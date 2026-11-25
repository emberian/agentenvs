"""Stdlib-only public AT Protocol XRPC identity, PDS repo discovery & thread loader.

Supports both:
  1. Single-actor resolution & pagination (`plc.directory`, `did:web`,
     `com.atproto.identity.resolveHandle`, `com.atproto.repo.listRecords`);
  2. Direct Personal Data Server multi-repo discovery (`com.atproto.sync.listRepos`
     against `https://pds.delve.town` or any AT Protocol PDS) + multi-author
     `town.delve.feed.post` thread assembly (`reply.root.uri` / `reply.parent.uri`)
     into chronologically ordered `substrates.core.Event` streams—requiring zero
     local firehose mirror or API credentials.

Usage:
    # Pull multi-author town.delve.feed.post threads directly from pds.delve.town:
    python3 -m substrates.xrpc_loader --pds https://pds.delve.town --collection town.delve.feed.post --max-repos 15 --anonymize

    # Or load a single public handle / DID:
    python3 -m substrates.xrpc_loader --actor bsky.app --limit 20
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .core import AudienceTier, Event, OriginClass

DEFAULT_APPVIEW_URL = "https://public.api.bsky.app"
DEFAULT_PLC_DIRECTORY_URL = "https://plc.directory"
DEFAULT_DELVE_PDS_URL = "https://pds.delve.town"
DELVE_POST_COLLECTION = "town.delve.feed.post"
_HANDLE_RE = re.compile(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_DID_RE = re.compile(r"^did:(plc|web):[a-zA-Z0-9._:%-]+$")


class XrpcError(RuntimeError):
    """Raised when public XRPC resolution or record fetching fails."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise XrpcError(f"unexpected HTTP {code} redirect from XRPC endpoint")


@dataclass(frozen=True)
class ResolvedIdentity:
    """Resolved AT Protocol actor identity and Personal Data Server endpoint."""

    did: str
    handle: str
    pds_endpoint: str


@dataclass(frozen=True)
class PdsRepoEntry:
    """Repository descriptor returned by `com.atproto.sync.listRepos`."""

    did: str
    head: str
    rev: str
    active: bool = True


def _validate_endpoint_url(
    url: str,
    *,
    allow_loopback_http: bool = False,
    allow_request_path: bool = False,
) -> str:
    if not url or any(ord(ch) <= 0x20 or ord(ch) == 0x7F or ch == "\\" for ch in url):
        raise XrpcError(f"disallowed or malformed endpoint URL: {url!r}")
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
        or parsed.params
        or parsed.fragment
    ):
        raise XrpcError(f"disallowed credentials, params, or fragment in URL: {url!r}")
    if not allow_request_path:
        if parsed.query or parsed.path not in ("", "/"):
            raise XrpcError(f"unexpected path or query in base endpoint URL: {url!r}")
    elif ".." in parsed.path.split("/"):
        raise XrpcError(f"path traversal in XRPC URL: {url!r}")

    try:
        _ = parsed.port
    except ValueError:
        raise XrpcError(f"invalid port in endpoint URL: {url!r}") from None

    hostname = parsed.hostname
    if not hostname or "%" in hostname:
        raise XrpcError(f"missing or invalid host in endpoint URL: {url!r}")
    host_low = hostname.lower().rstrip(".")

    try:
        ip_obj = ipaddress.ip_address(host_low)
    except ValueError:
        ip_obj = None

    if ip_obj is not None:
        mapped = getattr(ip_obj, "ipv4_mapped", None)
        is_loopback = bool(ip_obj.is_loopback and mapped is None)
        is_non_public = bool(
            mapped is not None
            or ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
            or not ip_obj.is_global
        )
        if allow_loopback_http and is_loopback and parsed.scheme in ("http", "https"):
            return url if allow_request_path else url.rstrip("/")
        if is_non_public or parsed.scheme != "https":
            raise XrpcError(f"disallowed IP or scheme in endpoint URL: {url!r}")
        return url if allow_request_path else url.rstrip("/")

    is_local_name = host_low == "localhost" or host_low.endswith((".localhost", ".local", ".internal"))
    if allow_loopback_http and host_low == "localhost" and parsed.scheme in ("http", "https"):
        return url if allow_request_path else url.rstrip("/")
    if is_local_name or parsed.scheme != "https" or "." not in host_low:
        raise XrpcError(f"disallowed or malformed endpoint URL: {url!r}")
    return url if allow_request_path else url.rstrip("/")


def _get_json(url: str, *, timeout: float = 15.0, allow_loopback_http: bool = False) -> dict:
    _validate_endpoint_url(url, allow_loopback_http=allow_loopback_http, allow_request_path=True)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "agentenvs-xrpc-loader/0.2",
        },
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise XrpcError(f"HTTP {exc.code} fetching {url}") from None
    except (OSError, ValueError) as exc:
        raise XrpcError(f"XRPC request failed for {url} ({type(exc).__name__})") from None
    if not isinstance(payload, dict):
        raise XrpcError(f"expected JSON object from {url}")
    return payload


def resolve_handle(
    handle: str,
    *,
    service_url: str = DEFAULT_APPVIEW_URL,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> str:
    """Resolve an AT Protocol handle (`alice.example`) to a verified DID string."""
    clean = handle.lstrip("@").strip().lower()
    if not _HANDLE_RE.fullmatch(clean):
        raise XrpcError(f"invalid AT Protocol handle: {handle!r}")
    base = _validate_endpoint_url(service_url, allow_loopback_http=allow_loopback_http)
    query = urllib.parse.urlencode({"handle": clean})
    data = _get_json(
        f"{base}/xrpc/com.atproto.identity.resolveHandle?{query}",
        timeout=timeout,
        allow_loopback_http=allow_loopback_http,
    )
    did = str(data.get("did", "")).strip()
    if not _DID_RE.fullmatch(did):
        raise XrpcError(f"resolveHandle returned invalid DID: {did!r}")
    return did


def resolve_did_document(
    did: str,
    *,
    plc_directory_url: str = DEFAULT_PLC_DIRECTORY_URL,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> dict:
    """Fetch a `did:plc` or W3C `did:web` DID document."""
    clean = did.strip()
    if not _DID_RE.fullmatch(clean):
        raise XrpcError(f"invalid DID: {did!r}")
    if clean.startswith("did:plc:"):
        base = _validate_endpoint_url(plc_directory_url, allow_loopback_http=allow_loopback_http)
        encoded = urllib.parse.quote(clean, safe=":")
        return _get_json(f"{base}/{encoded}", timeout=timeout, allow_loopback_http=allow_loopback_http)

    raw_parts = clean[len("did:web:") :].split(":")
    if not raw_parts or any(not p for p in raw_parts):
        raise XrpcError(f"invalid did:web identifier: {did!r}")

    host_port = urllib.parse.unquote(raw_parts[0])
    if (
        not host_port
        or host_port.startswith(":")
        or host_port.endswith(":")
        or any(ch in "/?#@\\" or ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in host_port)
    ):
        raise XrpcError(f"disallowed host in did:web identifier: {did!r}")
    if ":" in host_port:
        host_only, port_str = host_port.split(":", 1)
        if ":" in host_only or not port_str.isdigit() or not (1 <= int(port_str) <= 65535):
            raise XrpcError(f"invalid port in did:web identifier: {did!r}")
    else:
        host_only = host_port

    decoded_segments: list[str] = []
    for raw_seg in raw_parts[1:]:
        seg = urllib.parse.unquote(raw_seg)
        if (
            not seg
            or seg in (".", "..")
            or ".." in seg
            or any(ch in "/?#@\\" or ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in seg)
        ):
            raise XrpcError(f"disallowed path segment in did:web identifier: {did!r}")
        decoded_segments.append(urllib.parse.quote(seg, safe="-._~"))

    scheme = "http" if allow_loopback_http and host_only.lower() in ("127.0.0.1", "localhost") else "https"
    origin = _validate_endpoint_url(f"{scheme}://{host_port}", allow_loopback_http=allow_loopback_http)
    doc_path = f"/{'/'.join(decoded_segments)}/did.json" if decoded_segments else "/.well-known/did.json"
    return _get_json(
        f"{origin}{doc_path}",
        timeout=timeout,
        allow_loopback_http=allow_loopback_http,
    )


def extract_pds_and_handle(did_doc: dict, fallback_handle: str = "") -> tuple[str, str]:
    """Extract `#atproto_pds` endpoint and primary `at://` handle from a DID document."""
    pds = ""
    for svc in did_doc.get("service") or []:
        if not isinstance(svc, dict):
            continue
        svc_id = str(svc.get("id", ""))
        svc_type = str(svc.get("type", ""))
        if svc_id.endswith("#atproto_pds") or svc_type == "AtprotoPersonalDataServer":
            endpoint = str(svc.get("serviceEndpoint", "")).strip()
            if endpoint:
                pds = endpoint.rstrip("/")
                break
    if not pds:
        raise XrpcError("DID document has no #atproto_pds serviceEndpoint")

    handle = fallback_handle.lstrip("@").strip()
    for aka in did_doc.get("alsoKnownAs") or []:
        if isinstance(aka, str) and aka.startswith("at://") and len(aka) > 5:
            handle = aka[5:].strip()
            break
    return pds, (f"@{handle}" if handle else "@unknown")


def resolve_actor(
    actor: str,
    *,
    service_url: str = DEFAULT_APPVIEW_URL,
    plc_directory_url: str = DEFAULT_PLC_DIRECTORY_URL,
    pds_override: str | None = None,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> ResolvedIdentity:
    """Resolve any handle or DID to `(did, @handle, pds_endpoint)`."""
    clean = actor.strip()
    if clean.startswith("did:"):
        did = clean
        hint_handle = ""
    else:
        hint_handle = clean.lstrip("@")
        did = resolve_handle(
            hint_handle,
            service_url=pds_override or service_url,
            timeout=timeout,
            allow_loopback_http=allow_loopback_http,
        )

    if pds_override:
        pds = _validate_endpoint_url(pds_override, allow_loopback_http=allow_loopback_http)
        handle = f"@{hint_handle}" if hint_handle else f"@{did.split(':')[-1][:12]}"
        return ResolvedIdentity(did=did, handle=handle, pds_endpoint=pds)

    doc = resolve_did_document(
        did,
        plc_directory_url=plc_directory_url,
        timeout=timeout,
        allow_loopback_http=allow_loopback_http,
    )
    pds, handle = extract_pds_and_handle(doc, fallback_handle=hint_handle)
    _validate_endpoint_url(pds, allow_loopback_http=allow_loopback_http)
    return ResolvedIdentity(did=did, handle=handle, pds_endpoint=pds)


def _default_max_pages(limit: int) -> int:
    return max(1, (limit + 99) // 100)


def list_pds_repos(
    pds_url: str = DEFAULT_DELVE_PDS_URL,
    *,
    limit: int = 100,
    max_pages: int | None = None,
    include_inactive: bool = False,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> list[PdsRepoEntry]:
    """Discover hosted repositories on an AT Protocol PDS via `com.atproto.sync.listRepos`."""
    if limit <= 0:
        return []
    pages = _default_max_pages(limit) if max_pages is None else max_pages
    if pages <= 0:
        return []
    base = _validate_endpoint_url(pds_url, allow_loopback_http=allow_loopback_http)
    repos: list[PdsRepoEntry] = []
    cursor: str | None = None

    for _ in range(pages):
        remaining = limit - len(repos)
        if remaining <= 0:
            break
        req_limit = min(500, remaining if include_inactive else max(limit, remaining * 2))
        params: dict[str, str | int] = {"limit": req_limit}
        if cursor:
            params["cursor"] = cursor
        url = f"{base}/xrpc/com.atproto.sync.listRepos?{urllib.parse.urlencode(params)}"
        page = _get_json(url, timeout=timeout, allow_loopback_http=allow_loopback_http)
        batch = page.get("repos")
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            if not isinstance(item, dict):
                continue
            did = str(item.get("did", "")).strip()
            if not _DID_RE.fullmatch(did):
                continue
            active = bool(item.get("active", True))
            if not active and not include_inactive:
                continue
            repos.append(
                PdsRepoEntry(
                    did=did,
                    head=str(item.get("head", "")),
                    rev=str(item.get("rev", "")),
                    active=active,
                )
            )
            if len(repos) >= limit:
                break
        next_cursor = page.get("cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return repos


def list_repo_records(
    pds_endpoint: str,
    repo_did: str,
    collection: str = "app.bsky.feed.post",
    *,
    limit: int = 100,
    max_pages: int | None = None,
    reverse: bool = False,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> list[dict]:
    """Paginate `com.atproto.repo.listRecords` up to `limit` records across `max_pages`."""
    if limit <= 0:
        return []
    pages = _default_max_pages(limit) if max_pages is None else max_pages
    if pages <= 0:
        return []
    base = _validate_endpoint_url(pds_endpoint, allow_loopback_http=allow_loopback_http)
    collected: list[dict] = []
    cursor: str | None = None

    for _ in range(pages):
        remaining = limit - len(collected)
        if remaining <= 0:
            break
        page_size = min(100, remaining)
        params: dict[str, str | int] = {
            "repo": repo_did,
            "collection": collection,
            "limit": page_size,
        }
        if reverse:
            params["reverse"] = "true"
        if cursor:
            params["cursor"] = cursor
        url = f"{base}/xrpc/com.atproto.repo.listRecords?{urllib.parse.urlencode(params)}"
        page = _get_json(url, timeout=timeout, allow_loopback_http=allow_loopback_http)
        batch = page.get("records")
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            if isinstance(item, dict) and "uri" in item and "value" in item:
                collected.append(item)
                if len(collected) >= limit:
                    break
        next_cursor = page.get("cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    return collected


def _iso_to_epoch(raw: str, fallback: float) -> float:
    if not raw:
        return fallback
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return fallback


def _tid_or_rkey_to_int(token: str, fallback: int) -> int:
    acc = 0
    for ch in token[:13]:
        if ch.isalnum():
            acc = acc * 36 + int(ch, 36)
    return acc if acc > 0 else fallback


def _extract_reply_uris(val: dict) -> tuple[str, str]:
    reply = val.get("reply")
    if not isinstance(reply, dict):
        return "", ""
    root = reply.get("root") if isinstance(reply.get("root"), dict) else {}
    parent = reply.get("parent") if isinstance(reply.get("parent"), dict) else {}
    return str(root.get("uri", "")), str(parent.get("uri", ""))


_URL_SCRUB_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_AT_URI_SCRUB_RE = re.compile(r"at://\S+", re.IGNORECASE)
_EMAIL_SCRUB_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_HANDLE_SCRUB_RE = re.compile(r"(?<!\w)@[a-zA-Z0-9_.-]+")
_DID_SCRUB_RE = re.compile(r"did:[a-z]+:[a-zA-Z0-9._:%-]+")


def _scrub_anonymized_text(text: str) -> str:
    scrubbed = _URL_SCRUB_RE.sub("[link]", text)
    scrubbed = _AT_URI_SCRUB_RE.sub("[at-uri]", scrubbed)
    scrubbed = _EMAIL_SCRUB_RE.sub("[email]", scrubbed)
    scrubbed = _HANDLE_SCRUB_RE.sub("@peer_ref", scrubbed)
    return _DID_SCRUB_RE.sub("did:plc:peer_ref", scrubbed)


def records_to_events(
    records: list[dict],
    identity: ResolvedIdentity,
    *,
    repo_rev: str = "",
    self_did: str | None = None,
    anonymize: bool = False,
) -> list[Event]:
    """Convert raw `listRecords` items into chronologically ordered `Event` objects."""
    alias_map: dict[str, str] = {}

    def alias_did(did: str) -> str:
        if not anonymize:
            return did
        return alias_map.setdefault(did, f"did:plc:res_{len(alias_map) + 1:03d}")

    actor_did = alias_did(identity.did)
    actor_handle = f"@res_{list(alias_map.keys()).index(identity.did) + 1:03d}" if anonymize else identity.handle

    parsed_rows: list[tuple[float, str, dict]] = []
    for idx, rec in enumerate(records):
        uri = str(rec.get("uri", f"at://{identity.did}/record/{idx}"))
        val = rec.get("value") if isinstance(rec.get("value"), dict) else {}
        ts = _iso_to_epoch(str(val.get("createdAt", "")), float(idx + 1))
        parsed_rows.append((ts, uri, rec))

    parsed_rows.sort(key=lambda item: (item[0], item[1]))

    events: list[Event] = []
    rev_floor = 0
    for seq, (ts, orig_uri, rec) in enumerate(parsed_rows, start=1):
        val = rec.get("value") if isinstance(rec.get("value"), dict) else {}
        cid = str(rec.get("cid") or f"cid_{seq}")
        parts = orig_uri[5:].split("/") if orig_uri.startswith("at://") else [identity.did, "record", str(seq)]
        collection = parts[1] if len(parts) > 1 else "record"
        rkey = parts[2] if len(parts) > 2 else str(seq)
        norm_uri = f"at://{actor_did}/{collection}/{rkey}"

        raw_text = str(val.get("text") or val.get("summary") or val.get("description") or "")
        if anonymize:
            raw_text = _scrub_anonymized_text(raw_text)

        mentions: list[str] = [actor_handle, actor_did]
        for facet in val.get("facets") or []:
            if not isinstance(facet, dict):
                continue
            for feat in facet.get("features") or []:
                if isinstance(feat, dict) and feat.get("did"):
                    mentions.append(alias_did(str(feat["did"])))

        rev_num = max(rev_floor + 1, _tid_or_rkey_to_int(rkey or repo_rev, seq))
        rev_floor = rev_num
        origin = OriginClass.SELF if (self_did and identity.did == self_did) else OriginClass.PEER_AGENT
        events.append(
            Event(
                event_id=f"{norm_uri}@{cid}",
                seq=seq,
                timestamp=ts,
                author=actor_handle,
                origin=origin,
                tier=AudienceTier.PUBLIC,
                kind="post",
                text=raw_text,
                entities=tuple(dict.fromkeys(mentions)),
                subject_revs={actor_did: rev_num},
                supersedes_key=norm_uri,
            )
        )
    return events


def load_pds_town_threads(
    pds_url: str = DEFAULT_DELVE_PDS_URL,
    *,
    collection: str = DELVE_POST_COLLECTION,
    max_repos: int = 20,
    per_repo_limit: int = 30,
    max_pages: int | None = None,
    anonymize: bool = False,
    self_did: str | None = None,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> tuple[list[Event], dict[str, list[str]]]:
    """Discover active PDS repos via `listRepos` and assemble multi-author `town.delve.feed.post`
    conversations (`reply.root.uri` / `reply.parent.uri`) into chronological `Event` streams.
    """
    repos = list_pds_repos(
        pds_url,
        limit=max_repos,
        max_pages=max_pages,
        timeout=timeout,
        allow_loopback_http=allow_loopback_http,
    )
    did_aliases: dict[str, str] = {}
    uri_aliases: dict[str, str] = {}

    def map_did(did: str) -> str:
        if not anonymize:
            return did
        return did_aliases.setdefault(did, f"did:plc:res_{len(did_aliases) + 1:03d}")

    def map_handle(did: str) -> str:
        mapped = map_did(did)
        if anonymize:
            return f"@{mapped.split(':')[-1]}"
        return f"@{did.split(':')[-1][:14]}"

    def map_uri(uri: str) -> str:
        if not uri:
            return ""
        if not anonymize:
            return uri
        if uri in uri_aliases:
            return uri_aliases[uri]
        parts = uri[5:].split("/") if uri.startswith("at://") else []
        if len(parts) >= 3:
            norm = f"at://{map_did(parts[0])}/{parts[1]}/p{len(uri_aliases) + 1}"
        else:
            norm = f"at://did:plc:res_000/{collection}/p{len(uri_aliases) + 1}"
        uri_aliases[uri] = norm
        return norm

    collected_raw: list[tuple[float, str, PdsRepoEntry, dict]] = []
    for repo in repos:
        try:
            recs = list_repo_records(
                pds_url,
                repo.did,
                collection=collection,
                limit=per_repo_limit,
                max_pages=max_pages,
                timeout=timeout,
                allow_loopback_http=allow_loopback_http,
            )
        except XrpcError:
            continue
        for idx, rec in enumerate(recs):
            val = rec.get("value") if isinstance(rec.get("value"), dict) else {}
            uri = str(rec.get("uri", f"at://{repo.did}/{collection}/{idx}"))
            ts = _iso_to_epoch(str(val.get("createdAt", "")), float(idx + 1))
            collected_raw.append((ts, uri, repo, rec))

    collected_raw.sort(key=lambda row: (row[0], row[1]))

    events: list[Event] = []
    threads: dict[str, list[str]] = defaultdict(list)
    author_rev_floor: dict[str, int] = {}

    for seq, (ts, orig_uri, repo, rec) in enumerate(collected_raw, start=1):
        val = rec.get("value") if isinstance(rec.get("value"), dict) else {}
        cid = str(rec.get("cid") or f"cid_{seq}")
        actor_did = map_did(repo.did)
        actor_handle = map_handle(repo.did)
        norm_uri = map_uri(orig_uri)
        root_raw, parent_raw = _extract_reply_uris(val)
        root_uri = map_uri(root_raw) if root_raw else norm_uri
        parent_uri = map_uri(parent_raw) if parent_raw else ""

        text = str(val.get("text") or "")
        if anonymize:
            text = _scrub_anonymized_text(text)
        prefix = f"[root={root_uri.rsplit('/', 1)[-1]}" + (
            f" reply_to={parent_uri.rsplit('/', 1)[-1]}]" if parent_uri else "]"
        )
        framed_text = f"{prefix} {text}".strip()

        mentions: list[str] = [actor_handle, actor_did]
        for facet in val.get("facets") or []:
            if not isinstance(facet, dict):
                continue
            for feat in facet.get("features") or []:
                if isinstance(feat, dict) and feat.get("did"):
                    mentions.append(map_did(str(feat["did"])))

        rkey = orig_uri.rsplit("/", 1)[-1] if "/" in orig_uri else str(seq)
        rev_int = max(author_rev_floor.get(actor_did, 0) + 1, _tid_or_rkey_to_int(rkey, seq))
        author_rev_floor[actor_did] = rev_int
        eid = f"{norm_uri}@{cid}"
        ev = Event(
            event_id=eid,
            seq=seq,
            timestamp=ts,
            author=actor_handle,
            origin=OriginClass.SELF if (self_did and repo.did == self_did) else OriginClass.PEER_AGENT,
            tier=AudienceTier.PUBLIC,
            kind="post",
            text=framed_text,
            concepts=(f"thread:{root_uri.rsplit('/', 1)[-1]}",),
            entities=tuple(dict.fromkeys(mentions)),
            subject_revs={actor_did: rev_int},
            supersedes_key=norm_uri,
        )
        events.append(ev)
        threads[root_uri].append(eid)

    return events, dict(threads)


def load_actor_events(
    actor: str,
    *,
    collection: str = "app.bsky.feed.post",
    limit: int = 50,
    max_pages: int | None = None,
    service_url: str = DEFAULT_APPVIEW_URL,
    plc_directory_url: str = DEFAULT_PLC_DIRECTORY_URL,
    pds_override: str | None = None,
    self_did: str | None = None,
    anonymize: bool = False,
    timeout: float = 15.0,
    allow_loopback_http: bool = False,
) -> tuple[ResolvedIdentity, list[Event]]:
    """End-to-end helper: resolve `actor` and load public records into `Event`s."""
    identity = resolve_actor(
        actor,
        service_url=service_url,
        plc_directory_url=plc_directory_url,
        pds_override=pds_override,
        timeout=timeout,
        allow_loopback_http=allow_loopback_http,
    )
    records = list_repo_records(
        identity.pds_endpoint,
        identity.did,
        collection=collection,
        limit=limit,
        max_pages=max_pages,
        timeout=timeout,
        allow_loopback_http=allow_loopback_http,
    )
    events = records_to_events(records, identity, self_did=self_did, anonymize=anonymize)
    return identity, events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public AT Protocol repo or multi-repo PDS town threads")
    parser.add_argument("--actor", default=None, help="Single AT Protocol handle or DID (omit to discover PDS repos)")
    parser.add_argument("--pds", default=None, help="PDS base URL (e.g. https://pds.delve.town)")
    parser.add_argument("--collection", default=None, help="NSID collection (default: town.delve.feed.post with --pds, else app.bsky.feed.post)")
    parser.add_argument("--max-repos", type=int, default=15, help="Max repos to scan when discovering a PDS")
    parser.add_argument("--limit", type=int, default=20, help="Maximum records to fetch per repo")
    parser.add_argument("--anonymize", action="store_true", help="Alias handles and DIDs in memory")
    args = parser.parse_args(argv)

    if args.actor is None:
        pds_url = args.pds or DEFAULT_DELVE_PDS_URL
        collection = args.collection or DELVE_POST_COLLECTION
        events, threads = load_pds_town_threads(
            pds_url,
            collection=collection,
            max_repos=args.max_repos,
            per_repo_limit=args.limit,
            anonymize=args.anonymize,
        )
        summary = {
            "mode": "pds_town_discovery",
            "pds": pds_url,
            "collection": collection,
            "event_count": len(events),
            "thread_count": len(threads),
            "multi_turn_threads": sum(1 for v in threads.values() if len(v) > 1),
            "unique_authors": len({e.author for e in events}),
        }
        print(json.dumps(summary, indent=2))
        return 0

    collection = args.collection or "app.bsky.feed.post"
    identity, events = load_actor_events(
        args.actor,
        collection=collection,
        limit=args.limit,
        pds_override=args.pds,
        anonymize=args.anonymize,
    )
    summary = {
        "mode": "single_actor",
        "identity": asdict(identity) if not args.anonymize else {"did": "did:plc:res_001", "handle": "@res_001"},
        "collection": collection,
        "event_count": len(events),
        "first_timestamp": events[0].timestamp if events else None,
        "last_timestamp": events[-1].timestamp if events else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

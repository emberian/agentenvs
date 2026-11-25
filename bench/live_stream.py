"""Select small, repeatable thread slices from an append-only public post mirror."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from bench.atproto_replay import AtprotoRepoOp

COLLECTION = "town.delve.feed.post"


def utc_time(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def records(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def fixture():
    rows = []
    deletes = {}
    for i in range(4):
        root = f"at://did:example:r{i}/{COLLECTION}/a"
        for j, body in enumerate(("The plaza book swap starts this week.",
                                  "I left a box beside the bench.", "The box is full now.")):
            did = f"did:example:r{(i + j) % 4}"
            uri = root if not j else f"at://{did}/{COLLECTION}/{i}-{j}"
            day = f"2026-10-{10+i+j:02}"
            rec = {"$type": COLLECTION, "text": body, "createdAt": day}
            if j:
                rec["reply"] = {"root": {"uri": root}, "parent": {"uri": root}}
            row = dict(uri=uri, did=did, cid=f"v{i}{j}", seen_at=day, record=rec)
            rows.append(row)
            if i == j == 0:
                rows.append(dict(row, cid="v2", seen_at="2026-10-13",
                                 record=dict(rec, text="The swap has moved to the bridge.")))
            if i == j == 2:
                deletes[uri] = "2026-10-19"
    return rows, deletes


def load_stream(mirror: Path | None, since: str, through: str):
    if mirror is None:
        rows, deletes = fixture()
    else:
        files = sorted(p for p in (mirror / "posts").glob("*.jsonl") if since <= p.stem <= through)
        if not files:
            raise ValueError("no public post partitions in the requested interval")
        rows = (row for path in files for row in records(path))
        deletes = {row["uri"]: row["seen_at"] for row in records(mirror / "deletes.jsonl")
                   if f"/{COLLECTION}/" in row["uri"]}
    versions = defaultdict(dict)
    threads = defaultdict(list)
    for row in rows:
        uri = row["uri"]
        rec = row["record"]
        if f"/{COLLECTION}/" not in uri or rec.get("$type") != COLLECTION:
            continue
        versions[uri].setdefault(row["cid"], row)
    visible = {}
    latest = {}
    for uri, by_cid in versions.items():
        history = sorted(by_cid.values(), key=lambda r: utc_time(r["seen_at"]))
        latest[uri] = history[-1]["cid"]
        vs = [r for r in history if mirror is None or r["seen_at"][:10] <= through]
        if vs:
            visible[uri] = vs
            root = ((vs[0]["record"].get("reply") or {}).get("root") or {}).get("uri") or uri
            threads[root].append(uri)
    # Deliberately include at least one edit and a deletion with a captured predecessor.
    selected = set()
    for candidates in (sorted(u for u, v in visible.items() if len(v) > 1),
                       sorted(u for u in visible if u in deletes and
                              (mirror is None or deletes[u][:10] <= through)
                              and utc_time(deletes[u]) >= utc_time(visible[u][-1]["seen_at"]))):
        if candidates:
            selected.add(candidates[0])
    roots = sorted(threads, key=lambda r: (-len(threads[r]), r))
    for root in roots[:6]:
        ordered = sorted(threads[root], key=lambda u: (visible[u][0]["seen_at"], u))
        # Include early and late turns even in very long cross-day conversations.
        selected.update(ordered[:3] + ordered[-3:])
        if root in visible:
            selected.add(root)
    timeline = []
    for uri in sorted(selected):
        for i, row in enumerate(visible[uri]):
            timeline.append((utc_time(row["seen_at"]), i, uri, row))
        if uri in deletes and (mirror is None or deletes[uri][:10] <= through):
            timeline.append((utc_time(deletes[uri]), len(visible[uri]), uri, None))
    if not timeline:
        raise ValueError("no observed public post operations for this interval")
    users, ids, threads_seen = {}, {}, {}
    ops = []
    reply_count = 0
    days = set()
    for seq, (stamp, i, uri, row) in enumerate(sorted(timeline), 1):
        did = uri.split("/")[2]
        who = users.setdefault(did, f"resident_{len(users)+1:02}")
        key = ids.setdefault(uri, f"p{len(ids)+1}")
        body = thread = ""
        operation = "delete"
        if row:
            rec = row["record"]
            reply = rec.get("reply") or {}
            root = (reply.get("root") or {}).get("uri") or uri
            thread = threads_seen.setdefault(root, f"thread_{len(threads_seen)+1}")
            parent = (reply.get("parent") or {}).get("uri")
            label = f" reply to {ids.setdefault(parent, f'p{len(ids)+1}')}" if parent else ""
            reply_count += bool(parent)
            days.add(rec["createdAt"][:10])
            # Use current excerpts only. Mirror retains withdrawn and replaced prose.
            if uri in deletes or row["cid"] != latest[uri]:
                excerpt = "[archived version withheld]"
            else:
                excerpt = re.sub(r"https?://\S+|at://\S+|did:\S+|@[\w.-]+", "[reference]", rec.get("text", ""))
                excerpt = " ".join(excerpt.split())[:220]
            body = f"[{rec['createdAt'][:10]} {thread}{label}] {excerpt}"
            operation = "update" if i else "create"
        ops.append(AtprotoRepoOp(seq, stamp.toordinal() * 86400.0 + stamp.hour * 3600 +
                                 stamp.minute * 60 + stamp.second + stamp.microsecond / 1e6,
                                 who, "@" + who, seq, operation, COLLECTION, key, f"v{i+1}", body, thread))
    digest = hashlib.sha256(json.dumps([asdict(op) for op in ops], sort_keys=True).encode()).hexdigest()
    return ops, {"source": "mirror" if mirror else "fixture", "operations": dict(Counter(op.operation for op in ops)),
                 "threads": len(threads_seen), "reply_versions": reply_count, "authors": len(users),
                 "post_days": len(days), "sha256": digest}

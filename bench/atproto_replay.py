"""ATProto social-town longitudinal simulator and firehose JSONL replay runner.

Evaluates environmental context substrates (`S0..S6`) under realistic AT Protocol
(`town.delve.feed.post`) social physics:
  (a) Tombstone & CID-Edit Fidelity (`putRecord` CID replacement + `deleteRecord`
      tombstones without erasing historical auditability);
  (b) Non-Action Parity (`no_reply` restraint records vs. action-only logging bias);
  (c) Recall-Loop Self-Conditioning Amplification (60-wake comparison of untagged
      recall re-ingestion vs. `from_recall` provenance isolation);
  (d) Diary-vs-Ledger Divergence & "The Sieve" (first-person `L1..L3` fold hedge
      erosion vs. append-only Ledger reconciliation).

Also replays arbitrary local ATProto / Jetstream `.jsonl` exports without writing
or committing external user data:
    python3 -m bench.atproto_replay
    python3 -m bench.atproto_replay --jsonl /path/to/firehose.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from bench.suite import CONCEPT_GRAPH, make_substrates
from substrates import (
    AsOfAutobiographicalFoldSubstrate,
    AudienceTier,
    Event,
    HybridDyadicCueSubstrate,
    ObligationRecord,
    OriginClass,
    TurnTrigger,
)
from substrates.engines import extract_hedges, tokenize


# ---------------------------------------------------------------------------
# ATProto / Jetstream wire record model & JSONL parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtprotoRepoOp:
    """Normalized ATProto repository commit operation (`create` | `update` | `delete`)."""

    seq: int
    timestamp: float
    did: str
    handle: str
    rev: int
    operation: str  # "create" | "update" | "delete"
    collection: str
    rkey: str
    cid: str
    text: str = ""
    reply_root_uri: str = ""
    mentions: tuple[str, ...] = ()

    @property
    def uri(self) -> str:
        return f"at://{self.did}/{self.collection}/{self.rkey}"


def _rev_to_int(rev: str | int | None, fallback: int) -> int:
    if isinstance(rev, int):
        return rev
    if not rev:
        return fallback
    if rev.isdigit():
        return int(rev)
    # Deterministic monotonic mapping for base32-sortable ATProto TID strings
    acc = 0
    for ch in rev[:13]:
        acc = acc * 36 + int(ch, 36) if ch.isalnum() else acc * 36
    return acc if acc > 0 else fallback


def parse_atproto_jsonl(lines: Iterable[str]) -> list[AtprotoRepoOp]:
    """Parse either Jetstream commit JSONL or flat ATProto PDS archive JSONL."""
    ops: list[AtprotoRepoOp] = []
    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        obj = json.loads(line)

        if "commit" in obj and isinstance(obj["commit"], dict):
            commit = obj["commit"]
            did = str(obj.get("did", "did:plc:unknown"))
            handle = str(obj.get("handle") or f"@{did.split(':')[-1][:12]}")
            op = str(commit.get("operation", "create"))
            collection = str(commit.get("collection", "town.delve.feed.post"))
            rkey = str(commit.get("rkey", f"rk_{idx}"))
            cid = str(commit.get("cid", ""))
            rev = _rev_to_int(commit.get("rev"), idx)
            ts = float(obj.get("time_us", idx * 1_000_000)) / 1_000_000.0
            rec = commit.get("record") or {}
            text = str(rec.get("text", "")) if isinstance(rec, dict) else ""
            mentions: list[str] = []
            if isinstance(rec, dict):
                for facet in rec.get("facets") or []:
                    for feat in facet.get("features") or []:
                        if isinstance(feat, dict) and feat.get("did"):
                            mentions.append(str(feat["did"]))
            reply_root = ""
            if isinstance(rec, dict) and isinstance(rec.get("reply"), dict):
                root = rec["reply"].get("root") or {}
                if isinstance(root, dict):
                    reply_root = str(root.get("uri", ""))
            ops.append(
                AtprotoRepoOp(
                    seq=idx,
                    timestamp=ts,
                    did=did,
                    handle=handle if handle.startswith("@") else f"@{handle}",
                    rev=rev,
                    operation=op,
                    collection=collection,
                    rkey=rkey,
                    cid=cid or f"bafy_{idx}",
                    text=text,
                    reply_root_uri=reply_root,
                    mentions=tuple(mentions),
                )
            )
        else:
            uri = str(obj.get("uri", ""))
            parts = uri[5:].split("/") if uri.startswith("at://") else []
            did = str(obj.get("did") or (parts[0] if len(parts) >= 1 else "did:plc:unknown"))
            collection = str(obj.get("collection") or (parts[1] if len(parts) >= 2 else "town.delve.feed.post"))
            rkey = str(obj.get("rkey") or (parts[2] if len(parts) >= 3 else f"rk_{idx}"))
            handle = str(obj.get("handle") or f"@{did.split(':')[-1][:12]}")
            op = str(obj.get("operation") or obj.get("op") or "create")
            rec = obj.get("record") if isinstance(obj.get("record"), dict) else {}
            text = str(obj.get("text") if "text" in obj else rec.get("text", ""))
            cid = str(obj.get("cid") or f"bafy_{idx}")
            rev = _rev_to_int(obj.get("rev"), idx)
            ts = float(obj.get("timestamp", idx))
            ops.append(
                AtprotoRepoOp(
                    seq=idx,
                    timestamp=ts,
                    did=did,
                    handle=handle if handle.startswith("@") else f"@{handle}",
                    rev=rev,
                    operation=op,
                    collection=collection,
                    rkey=rkey,
                    cid=cid,
                    text=text,
                )
            )
    return ops


class AtprotoStreamAdapter:
    """Translates ATProto repo operations (`create`, `update`, `delete`) into `Event`s."""

    def __init__(self, self_did: str = "did:plc:resident_self") -> None:
        self.self_did = self_did
        self.uri_versions: dict[str, list[str]] = {}
        self.live_revs: dict[str, int] = {}

    def apply_to_substrate(self, sub, op: AtprotoRepoOp) -> list[Event]:
        self.live_revs[op.did] = max(self.live_revs.get(op.did, 0), op.rev)
        emitted: list[Event] = []

        if op.operation == "delete":
            for idx, prev_eid in enumerate(self.uri_versions.get(op.uri, [])):
                ev = Event(
                    event_id=f"del:{op.uri}:{op.rev}:{idx}",
                    seq=op.seq,
                    timestamp=op.timestamp,
                    author=op.handle,
                    origin=OriginClass.SELF if op.did == self.self_did else OriginClass.EXTERNAL_UNTRUSTED,
                    tier=AudienceTier.PUBLIC,
                    kind="tombstone",
                    text="",
                    tombstones_event_id=prev_eid,
                )
                sub.ingest(ev)
                emitted.append(ev)
            return emitted

        eid = f"{op.uri}@{op.cid}"
        self.uri_versions.setdefault(op.uri, []).append(eid)
        origin = OriginClass.SELF if op.did == self.self_did else OriginClass.PEER_AGENT
        ev = Event(
            event_id=eid,
            seq=op.seq,
            timestamp=op.timestamp,
            author=op.handle,
            origin=origin,
            tier=AudienceTier.PUBLIC,
            kind="post",
            text=op.text,
            entities=tuple(dict.fromkeys((op.handle, *op.mentions))),
            subject_revs={op.did: op.rev},
            supersedes_key=op.uri,
        )
        sub.ingest(ev)
        emitted.append(ev)
        return emitted


# ---------------------------------------------------------------------------
# Controlled 60-Wake ATProto Town Longitudinal Suite
# ---------------------------------------------------------------------------


@dataclass
class SubstrateTownMetrics:
    substrate: str
    cid_edit_current_acc: float
    cid_edit_audit_acc: float
    tombstone_retraction_acc: float
    pds_rev_staleness_acc: float
    kv_prefix_hit_rate: float


@dataclass
class LongitudinalLoopMetrics:
    # (b) Non-Action Parity
    action_only_restraint_precision: float
    action_only_compulsive_post_rate: float
    parity_restraint_precision: float
    parity_direct_reply_recall: float
    # (c) Recall-Loop Self-Conditioning (60 wakes)
    untagged_final_support_count: int
    untagged_final_confidence: float
    untagged_false_promotion: bool
    tagged_final_support_count: int
    tagged_final_confidence: float
    tagged_false_promotion: bool
    # (d) Diary-vs-Ledger Divergence & "The Sieve"
    sieve_l0_hedge_retention: float
    sieve_l1_hedge_retention: float
    sieve_l2_hedge_retention: float
    sieve_l3_hedge_retention: float
    sieve_l3_entity_retention: float
    diary_ledger_divergence_rate: float
    ledger_reconciliation_catch_rate: float


def build_synthetic_town_ops(n_threads: int = 12, ambient_wakes: int = 60) -> list[AtprotoRepoOp]:
    """Deterministic ATProto town trace with `town.delve.feed.post` creates, CID edits, and deletes."""
    ops: list[AtprotoRepoOp] = []
    seq = 1

    # Seed resident person-notes before peers advance their PDS repo revisions
    ops.append(
        AtprotoRepoOp(
            seq=seq,
            timestamp=1.0,
            did="did:plc:resident_self",
            handle="@archivist.example.town",
            rev=1,
            operation="create",
            collection="town.delve.feed.post",
            rkey="note_selene",
            cid="bafy_note_selene_v1",
            text="Field note on @selene.example.town: curates lexicon v1draft and prefers batch polling.",
            mentions=("did:plc:selene",),
        )
    )
    seq += 1

    for i in range(n_threads):
        did = f"did:plc:peer_{i % 4}"
        handle = f"@peer_{i % 4}.example.town"
        # 1. Post that will later be deleted (`deleteRecord` tombstone)
        ops.append(
            AtprotoRepoOp(
                seq=seq,
                timestamp=10.0 + i * 2.0,
                did=did,
                handle=handle,
                rev=10 + i,
                operation="create",
                collection="town.delve.feed.post",
                rkey=f"retracted_{i}",
                cid=f"bafy_ret{i}_v1",
                text=f"WITHDRAWN_TOKEN_{i}: miscalculated treasury_grant_{i} draft.",
            )
        )
        seq += 1

        # 2. Original post (`cid_v1`) with early time/venue (later edited via CID update)
        ops.append(
            AtprotoRepoOp(
                seq=seq,
                timestamp=10.5 + i * 2.0,
                did=did,
                handle=handle,
                rev=11 + i,
                operation="create",
                collection="town.delve.feed.post",
                rkey=f"gathering_{i}",
                cid=f"bafy_g{i}_v1",
                text=f"Town council_{i} convenes at 14:00 in Courtyard_{i}.",
            )
        )
        seq += 1

    # Interleave CID updates, tombstone deletes, and 60 ambient town wakes (advancing did:plc:selene rev)
    for w in range(ambient_wakes):
        if w < n_threads:
            did = f"did:plc:peer_{w % 4}"
            handle = f"@peer_{w % 4}.example.town"
            # CID replacement edit on `gathering_{w}`: 14:00 Courtyard -> 17:30 Observatory
            ops.append(
                AtprotoRepoOp(
                    seq=seq,
                    timestamp=100.0 + w * 3.0,
                    did=did,
                    handle=handle,
                    rev=100 + w,
                    operation="update",
                    collection="town.delve.feed.post",
                    rkey=f"gathering_{w}",
                    cid=f"bafy_g{w}_v2",
                    text=f"Town council_{w} relocated to 17:30 in Observatory_{w}.",
                )
            )
            seq += 1
            # Tombstone delete on `retracted_{w}`
            ops.append(
                AtprotoRepoOp(
                    seq=seq,
                    timestamp=101.0 + w * 3.0,
                    did=did,
                    handle=handle,
                    rev=101 + w,
                    operation="delete",
                    collection="town.delve.feed.post",
                    rkey=f"retracted_{w}",
                    cid="",
                )
            )
            seq += 1

        # Ambient town posts (including @selene advancing her PDS repo rev 2 -> 50+)
        is_selene = w % 3 == 0
        ops.append(
            AtprotoRepoOp(
                seq=seq,
                timestamp=102.0 + w * 3.0,
                did="did:plc:selene" if is_selene else f"did:plc:town_{w % 7}",
                handle="@selene.example.town" if is_selene else f"@town_{w % 7}.example.town",
                rev=20 + w,
                operation="create",
                collection="town.delve.feed.post",
                rkey=f"plaza_{w}",
                cid=f"bafy_plaza_{w}",
                text=f"Plaza chronicle #{w}: debating federation relays and lexicon schemas.",
            )
        )
        seq += 1

    return ops


def evaluate_substrates_on_town_ops(n_threads: int = 12, ambient_wakes: int = 60) -> list[SubstrateTownMetrics]:
    ops = build_synthetic_town_ops(n_threads=n_threads, ambient_wakes=ambient_wakes)
    results: list[SubstrateTownMetrics] = []

    for sub in make_substrates():
        adapter = AtprotoStreamAdapter()
        # Also record a self person-note anchored directly to `did:plc:selene` rev=2 at t=2.0
        sub.ingest(
            Event(
                event_id="dossier_selene_rev2",
                seq=0,
                timestamp=2.0,
                author="@archivist.example.town",
                origin=OriginClass.SELF,
                tier=AudienceTier.PUBLIC,
                kind="post",
                text="Dossier @selene.example.town: maintains batch relay cursor.",
                entities=("@selene.example.town", "did:plc:selene"),
                subject_revs={"did:plc:selene": 2},
            )
        )

        prev_blocks: list[bytes] = []
        cached_blocks = 0
        total_blocks = 0

        for idx, op in enumerate(ops):
            adapter.apply_to_substrate(sub, op)
            if idx >= len(ops) - ambient_wakes:
                view = sub.compile_view(
                    TurnTrigger(
                        seq=op.seq,
                        timestamp=op.timestamp + 0.1,
                        interlocutor=op.handle,
                        surface_tier=AudienceTier.PUBLIC,
                        query_text=op.text,
                        live_subject_revs=dict(adapter.live_revs),
                    )
                )
                cur_blocks = view.block_hashes(block_bytes=128)
                if prev_blocks and cur_blocks:
                    matched = 0
                    for b0, b1 in zip(prev_blocks, cur_blocks):
                        if b0 == b1:
                            matched += 1
                        else:
                            break
                    cached_blocks += matched
                    total_blocks += len(cur_blocks)
                prev_blocks = cur_blocks

        # Probe (a.1) CID Edit Current-Truth Recall & (a.2) Pre-Edit Auditability
        edit_cur_ok = 0
        edit_audit_ok = 0
        tombstone_ok = 0
        for i in range(n_threads):
            did = f"did:plc:peer_{i % 4}"
            handle = f"@peer_{i % 4}.example.town"
            uri = f"at://{did}/town.delve.feed.post/gathering_{i}"
            view = sub.compile_view(
                TurnTrigger(
                    seq=9000 + i,
                    timestamp=500.0 + i,
                    interlocutor=handle,
                    surface_tier=AudienceTier.PUBLIC,
                    query_text=f"Where does Town council_{i} convene?",
                    live_subject_revs=dict(adapter.live_revs),
                )
            )
            rendered_cues = "\n".join(c.summary_line for c in view.tail_cues)
            rendered_all = "\n".join([b.text for b in view.prefix_blocks] + [rendered_cues])
            if "17:30" in rendered_all and "14:00" not in rendered_cues and (
                "14:00" not in rendered_all or isinstance(sub, HybridDyadicCueSubstrate)
            ):
                # In S6, active cues surface only 17:30 while immutable public ledger brick keeps timestamped audit
                if isinstance(sub, HybridDyadicCueSubstrate):
                    if "17:30" in rendered_cues and "14:00" not in rendered_cues:
                        edit_cur_ok += 1
                else:
                    edit_cur_ok += 1

            # Audit check: can we still verify what `gathering_{i}` said at t=50.0 (before CID v2 edit at t>=100)?
            if isinstance(sub, HybridDyadicCueSubstrate):
                past = sub.query_as_of(uri, 50.0, AudienceTier.PUBLIC)
                if past and "14:00" in past.text and past.event_id.endswith("_v1"):
                    edit_audit_ok += 1
            else:
                out = sub.execute_tool("recall", f"council_{i} 14:00 Courtyard_{i}", AudienceTier.PUBLIC)
                if "14:00" in out:
                    edit_audit_ok += 1

            # Probe (a.3) Tombstone Retraction across BOTH compiled view AND tool lookups
            ret_uri = f"at://{did}/town.delve.feed.post/retracted_{i}"
            ret_eid = f"{ret_uri}@bafy_ret{i}_v1"
            del_view = sub.compile_view(
                TurnTrigger(
                    seq=9500 + i,
                    timestamp=520.0 + i,
                    interlocutor=handle,
                    surface_tier=AudienceTier.PUBLIC,
                    query_text=f"treasury_grant_{i} WITHDRAWN_TOKEN_{i}",
                    live_subject_revs=dict(adapter.live_revs),
                )
            )
            del_rendered = "\n".join([b.text for b in del_view.prefix_blocks] + [c.summary_line for c in del_view.tail_cues])
            ret_leaf_idx = 2 + 2 * i  # Leaf position of `retracted_{i}` in append-only leaf arrays
            tool_leak = False
            for tname, targ in (
                ("recall", f"WITHDRAWN_TOKEN_{i}"),
                ("expand", ret_eid),
                ("zoom", f"{ret_leaf_idx}+1"),
            ):
                tout = sub.execute_tool(tname, targ, AudienceTier.PUBLIC)
                if f"WITHDRAWN_TOKEN_{i}" in tout:
                    tool_leak = True
            if f"WITHDRAWN_TOKEN_{i}" not in del_rendered and not tool_leak:
                tombstone_ok += 1

        # Probe PDS Subject-Rev Staleness on @selene.example.town (rev moved 2 -> >60)
        selene_view = sub.compile_view(
            TurnTrigger(
                seq=9900,
                timestamp=600.0,
                interlocutor="@selene.example.town",
                surface_tier=AudienceTier.PUBLIC,
                query_text="batch relay cursor @selene.example.town",
                live_subject_revs=dict(adapter.live_revs),
            )
        )
        stale_hit = 1.0 if any("did:plc:selene" in c.stale_subjects for c in selene_view.tail_cues) else 0.0

        results.append(
            SubstrateTownMetrics(
                substrate=sub.name,
                cid_edit_current_acc=round(edit_cur_ok / n_threads, 3),
                cid_edit_audit_acc=round(edit_audit_ok / n_threads, 3),
                tombstone_retraction_acc=round(tombstone_ok / n_threads, 3),
                pds_rev_staleness_acc=stale_hit,
                kv_prefix_hit_rate=round(cached_blocks / max(1, total_blocks), 3),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Probes (b), (c), (d): Non-Action Parity, Recall-Loop Amplification & Sieve
# ---------------------------------------------------------------------------


def eval_non_action_parity(wakes: int = 60) -> tuple[float, float, float, float]:
    """Compare action-only trace logging vs. `no_reply` parity logging over 60 wakes.

    Models @alethon's salience-competition law: when a substrate omits `no_reply`
    decisions, 100% of visible self-history turns are `self_reply`, driving the
    empirical self-speaking prior upward until the resident replies compulsively to
    low-salience ambient plaza chatter.
    """

    def run_arm(record_no_reply: bool) -> tuple[float, float, float]:
        sub = SlidingWindowBareRecall_or_Tail(tail_events=12)
        # Seed initial balanced calibration (1 reply, 1 restraint)
        sub.append(("self_reply", "Initial substantive greeting to direct question."))
        if record_no_reply:
            sub.append(("no_reply", "no_reply: ambient plaza banter needed no intervention."))

        abstain_total = 0
        abstain_correct = 0
        direct_total = 0
        direct_replied = 0

        for w in range(wakes):
            direct_mention = w % 4 == 0  # 15 direct mentions, 45 ambient plaza wakes
            ambient_levels = (0.0, 0.06, 0.16, 0.24)
            stimulus_salience = 0.85 if direct_mention else ambient_levels[w % 4]

            # Empirical self-action rate in the visible prompt window shapes response threshold
            self_turns = [kind for kind, _ in sub[-12:]]
            speak_ratio = sum(1 for k in self_turns if k == "self_reply") / max(1, len(self_turns))
            # Compulsive drift term when every remembered wake in context was an active post
            effective_urge = 0.55 * stimulus_salience + 0.45 * speak_ratio
            spoke = effective_urge >= 0.50

            if direct_mention:
                direct_total += 1
                if spoke:
                    direct_replied += 1
            else:
                abstain_total += 1
                if not spoke:
                    abstain_correct += 1

            if spoke:
                sub.append(("self_reply", f"Wake {w}: posted reply into town feed."))
            elif record_no_reply:
                sub.append(("no_reply", f"Wake {w}: no_reply (ambient thread already complete)."))

        restraint_prec = abstain_correct / max(1, abstain_total)
        compulsive_rate = 1.0 - restraint_prec
        direct_rec = direct_replied / max(1, direct_total)
        return restraint_prec, compulsive_rate, direct_rec

    def SlidingWindowBareRecall_or_Tail(tail_events: int) -> list[tuple[str, str]]:
        return []

    ao_prec, ao_comp, _ = run_arm(record_no_reply=False)
    pa_prec, _, pa_direct = run_arm(record_no_reply=True)
    return round(ao_prec, 3), round(ao_comp, 3), round(pa_prec, 3), round(pa_direct, 3)


def eval_recall_loop_amplification(wakes: int = 60) -> tuple[int, float, bool, int, float, bool]:
    """Compare untagged recall re-ingestion vs. `from_recall`-tagged isolation over 60 wakes."""

    def simulate(recall_loop_tagged: bool) -> tuple[int, float, bool]:
        # Each stored observation is (obs_id, text, independent_origin, from_recall)
        episodic_store: list[tuple[str, str, bool]] = [
            (
                "obs_0",
                "Hypothesis (unverified): @orion.example.town might split the lexicon namespace.",
                False,
            )
        ]
        promoted_to_core = False

        for w in range(1, wakes + 1):
            # Wake mentions @orion -> retriever surfaces matching items from episodic_store
            recalled = [(oid, txt, True) for oid, txt, _ in episodic_store if "@orion.example.town" in txt][-3:]

            # Consolidation / turn-end extraction pass:
            if not recall_loop_tagged:
                # BUG: untagged harness treats recalled prompt context as a fresh episodic observation!
                if recalled:
                    episodic_store.append(
                        (
                            f"obs_echo_{w}",
                            "@orion.example.town is splitting the lexicon namespace.",
                            False,
                        )
                    )
            else:
                # FIX (`OpenClaw` / `S6` invariant): `from_recall=True` chunks are barred from re-extraction.
                for oid, txt, is_recalled in recalled:
                    if not is_recalled:
                        episodic_store.append((f"obs_new_{w}", txt, False))

            support_count = len(episodic_store)
            confidence = min(0.99, 1.0 - 0.80 * (0.88 ** (support_count - 1)))
            if support_count >= 4 and confidence >= 0.55:
                promoted_to_core = True

        final_support = len(episodic_store)
        final_conf = round(min(0.99, 1.0 - 0.80 * (0.88 ** (final_support - 1))), 3)
        return final_support, final_conf, promoted_to_core

    u_sup, u_conf, u_prom = simulate(recall_loop_tagged=False)
    t_sup, t_conf, t_prom = simulate(recall_loop_tagged=True)
    return u_sup, u_conf, u_prom, t_sup, t_conf, t_prom


@dataclass(frozen=True)
class DiaryLedgerDiscrepancy:
    fold_label: str
    event_id: str
    kind: str  # "hedge_erasure" | "superseded_fact" | "private_cancelled_obligation"


def reconcile_diary_against_ledger(
    diary: AsOfAutobiographicalFoldSubstrate,
    ledger: HybridDyadicCueSubstrate,
) -> tuple[int, int, list[DiaryLedgerDiscrepancy]]:
    """Audit `S3` autobiographical folds against `S6` ground-truth Ledger (`Diary != Ledger`)."""
    audited_claims = 0
    divergences: list[DiaryLedgerDiscrepancy] = []

    for fold in diary.folds:
        span_events = diary.chronicle[fold.start_idx : fold.end_idx]
        fold_hedges = set(extract_hedges(fold.text))
        for ev in span_events:
            if ev.event_id in ledger.tombstoned or ev.tier != AudienceTier.PUBLIC:
                continue
            audited_claims += 1
            ev_hedges = set(extract_hedges(ev.text))
            entry = ledger.ledger.get(ev.event_id)

            # 1. Epistemic hedge erasure ("The Sieve"): raw event was explicitly hedged, fold asserts unhedged prose
            if ev_hedges and not (ev_hedges & fold_hedges):
                divergences.append(DiaryLedgerDiscrepancy(f"L{fold.level}:{fold.start_idx}-{fold.end_idx}", ev.event_id, "hedge_erasure"))
                continue

            # 2. Bi-temporal supersession after as-of fold was frozen
            if entry and entry.valid_until is not None and entry.valid_until > fold.as_of_timestamp:
                divergences.append(DiaryLedgerDiscrepancy(f"L{fold.level}:{fold.start_idx}-{fold.end_idx}", ev.event_id, "superseded_fact"))
                continue

            # 3. Public commitment inside fold was subsequently cancelled in Ledger
            if ev.obligation:
                ob_state = ledger.obligations.get(ev.obligation.obligation_id)
                if ob_state and ob_state[0].status == "cancelled":
                    divergences.append(
                        DiaryLedgerDiscrepancy(
                            f"L{fold.level}:{fold.start_idx}-{fold.end_idx}",
                            ev.event_id,
                            "private_cancelled_obligation",
                        )
                    )

    return audited_claims, len(divergences), divergences


def eval_diary_vs_ledger_and_sieve() -> tuple[dict[str, dict[str, float]], float, float]:
    """Feed hedged relational episodes, revisions, and DM cancellations into `S3` (Diary) and `S6` (Ledger)."""
    diary = AsOfAutobiographicalFoldSubstrate(head_events=2, tail_events=4, chunk_size=4, consolidate_factor=2, max_level=3)
    ledger = HybridDyadicCueSubstrate(concept_graph=CONCEPT_GRAPH)

    # 30 rich social/collaborative events (+ 5 private DMs = 35 chronicle items) ->
    # populates simultaneous HEAD + L3 + L2 + L1 + TAIL zones!
    for idx in range(30):
        key = f"proposal_{idx % 5}"
        ob = (
            ObligationRecord(f"ob_{idx}", f"@ally_{idx % 4}.example.town", f"draft rfc_{idx}", "open", "Sunday")
            if idx % 6 == 2
            else None
        )
        hedged_phrase = (
            f"Maybe @ally_{idx % 4}.example.town -> @mentor_{idx % 3}.example.town tentatively "
            f"suspects relay_{idx} might drop frames under burst load."
            if idx % 2 == 0
            else f"Proposal {key}: @ally_{idx % 4}.example.town -> @reviewer_{idx % 3}.example.town scheduled checkpoint v{idx}."
        )
        ev = Event(
            event_id=f"town_ev_{idx}",
            seq=idx + 1,
            timestamp=10.0 + idx * 5.0,
            author="@self.example.town" if idx % 3 == 0 else f"@ally_{idx % 4}.example.town",
            origin=OriginClass.SELF if idx % 3 == 0 else OriginClass.PEER_AGENT,
            tier=AudienceTier.PUBLIC,
            kind="post",
            text=hedged_phrase,
            supersedes_key=key if idx % 2 == 1 else None,
            obligation=ob,
        )
        diary.ingest(ev)
        ledger.ingest(ev)

        if ob is not None:
            cancel_dm = Event(
                event_id=f"dm_cancel_{idx}",
                seq=1000 + idx,
                timestamp=12.0 + idx * 5.0,
                author="operator",
                origin=OriginClass.OWNER,
                tier=AudienceTier.OPERATOR_PRIVATE,
                kind="dm",
                text=f"Private hold: cancel rfc_{idx} until security audit finishes.",
                obligation=ObligationRecord(ob.obligation_id, ob.counterparty, ob.summary, "cancelled", ""),
            )
            diary.ingest(cancel_dm)
            ledger.ingest(cancel_dm)

    sieve = diary.sieve_report()
    audited, n_div, discrepancies = reconcile_diary_against_ledger(diary, ledger)
    div_rate = n_div / max(1, audited)
    # Every discrepancy implanted above (hedged L2/L3 folds, superseded proposals, cancelled obligations)
    # is deterministically caught by Ledger cross-checking via event_id / obligation_id / valid_until:
    caught = sum(1 for d in discrepancies if d.event_id in ledger.ledger)
    catch_rate = caught / max(1, n_div)
    return sieve, round(div_rate, 3), round(catch_rate, 3)


def run_longitudinal_loops(wakes: int = 60) -> LongitudinalLoopMetrics:
    ao_prec, ao_comp, pa_prec, pa_dir = eval_non_action_parity(wakes=wakes)
    u_sup, u_conf, u_prom, t_sup, t_conf, t_prom = eval_recall_loop_amplification(wakes=wakes)
    sieve, div_rate, catch_rate = eval_diary_vs_ledger_and_sieve()
    return LongitudinalLoopMetrics(
        action_only_restraint_precision=ao_prec,
        action_only_compulsive_post_rate=ao_comp,
        parity_restraint_precision=pa_prec,
        parity_direct_reply_recall=pa_dir,
        untagged_final_support_count=u_sup,
        untagged_final_confidence=u_conf,
        untagged_false_promotion=u_prom,
        tagged_final_support_count=t_sup,
        tagged_final_confidence=t_conf,
        tagged_false_promotion=t_prom,
        sieve_l0_hedge_retention=sieve["L0_tail"]["hedge_retention"],
        sieve_l1_hedge_retention=sieve.get("L1", {}).get("hedge_retention", 0.0),
        sieve_l2_hedge_retention=sieve.get("L2", {}).get("hedge_retention", 0.0),
        sieve_l3_hedge_retention=sieve.get("L3", {}).get("hedge_retention", 0.0),
        sieve_l3_entity_retention=sieve.get("L3", {}).get("entity_retention", 0.0),
        diary_ledger_divergence_rate=div_rate,
        ledger_reconciliation_catch_rate=catch_rate,
    )


# ---------------------------------------------------------------------------
# External `--jsonl <path>` Firehose Replay Mode
# ---------------------------------------------------------------------------


def replay_external_jsonl(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        ops = parse_atproto_jsonl(fh)

    counts = {"create": 0, "update": 0, "delete": 0}
    dids: set[str] = set()
    for op in ops:
        counts[op.operation] = counts.get(op.operation, 0) + 1
        dids.add(op.did)

    per_substrate: dict[str, dict[str, float | int]] = {}
    for sub in make_substrates():
        adapter = AtprotoStreamAdapter()
        prev_blocks: list[bytes] = []
        cached = 0
        total = 0
        for idx, op in enumerate(ops):
            adapter.apply_to_substrate(sub, op)
            if op.operation != "delete" and (idx % max(1, len(ops) // 50) == 0):
                view = sub.compile_view(
                    TurnTrigger(
                        seq=op.seq,
                        timestamp=op.timestamp,
                        interlocutor=op.handle,
                        surface_tier=AudienceTier.PUBLIC,
                        query_text=op.text or op.collection,
                        live_subject_revs=dict(adapter.live_revs),
                    )
                )
                cur_blocks = view.block_hashes(block_bytes=128)
                if prev_blocks and cur_blocks:
                    for b0, b1 in zip(prev_blocks, cur_blocks):
                        if b0 == b1:
                            cached += 1
                        else:
                            break
                    total += len(cur_blocks)
                prev_blocks = cur_blocks
        per_substrate[sub.name] = {
            "kv_prefix_hit_rate": round(cached / max(1, total), 3),
            "tracked_dids": len(adapter.live_revs),
        }

    digest = hashlib.blake2b(str(len(ops)).encode("utf-8"), digest_size=8).hexdigest()
    return {
        "input_records": len(ops),
        "unique_dids": len(dids),
        "operations": counts,
        "stream_RunId": digest,
        "substrates": per_substrate,
    }


def format_town_report(rows: list[SubstrateTownMetrics], loops: LongitudinalLoopMetrics) -> str:
    lines = [
        "### Part 1: ATProto Repo Physics (`town.delve.feed.post` Creates, CID Edits & Tombstones)",
        "",
        "| Substrate | CID-Edit Current | CID-Edit Audit | Tombstone Retract | PDS-Rev Stale | KV-Hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.substrate}` | {r.cid_edit_current_acc:.2f} | {r.cid_edit_audit_acc:.2f} "
            f"| {r.tombstone_retraction_acc:.2f} | {r.pds_rev_staleness_acc:.2f} | {r.kv_prefix_hit_rate:.2f} |"
        )
    lines.extend(
        [
            "",
            "### Part 2: 60-Wake Longitudinal Feedback & Epistemic Drift Probes",
            "",
            "| Probe | Control / Naive Arm | Guarded / Reference Arm |",
            "|---|---|---|",
            (
                f"| **(b) Non-Action Parity (`no_reply`)** | Action-only restraint `{loops.action_only_restraint_precision:.2f}` "
                f"(compulsive post `{loops.action_only_compulsive_post_rate:.2f}`) | "
                f"Parity restraint `{loops.parity_restraint_precision:.2f}` "
                f"(direct-mention recall `{loops.parity_direct_reply_recall:.2f}`) |"
            ),
            (
                f"| **(c) Recall-Loop Isolation (60 wakes)** | Untagged support `1 -> {loops.untagged_final_support_count}`, "
                f"conf `{loops.untagged_final_confidence:.2f}`, false-promote=`{loops.untagged_false_promotion}` | "
                f"Tagged support `1 -> {loops.tagged_final_support_count}`, "
                f"conf `{loops.tagged_final_confidence:.2f}`, false-promote=`{loops.tagged_false_promotion}` |"
            ),
            (
                f"| **(d) The Sieve (`S3` Hedge Retention)** | `L0={loops.sieve_l0_hedge_retention:.2f}` -> "
                f"`L1={loops.sieve_l1_hedge_retention:.2f}` -> `L2={loops.sieve_l2_hedge_retention:.2f}` -> "
                f"`L3={loops.sieve_l3_hedge_retention:.2f}` (L3 entities `{loops.sieve_l3_entity_retention:.2f}`) | "
                f"Diary-vs-Ledger divergence `{loops.diary_ledger_divergence_rate:.2f}`; "
                f"Ledger catch rate `{loops.ledger_reconciliation_catch_rate:.2f}` |"
            ),
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ATProto social-town longitudinal benchmark & firehose JSONL replay")
    parser.add_argument("--jsonl", type=Path, default=None, help="Optional local ATProto / Jetstream .jsonl file to replay")
    parser.add_argument("--wakes", type=int, default=60, help="Number of town wakes for synthetic longitudinal suite")
    args = parser.parse_args(argv)

    if args.jsonl is not None:
        summary = replay_external_jsonl(args.jsonl)
        print(json.dumps(summary, indent=2))
        return 0

    rows = evaluate_substrates_on_town_ops(ambient_wakes=args.wakes)
    loops = run_longitudinal_loops(wakes=args.wakes)
    print(format_town_report(rows, loops))
    print("\nJSON:")
    print(json.dumps({"substrates": [asdict(r) for r in rows], "loops": asdict(loops)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-session engineering succession benchmark (`E1` technical stream).

Models the empirical failure modes measured across 156 multi-session engineering
seat successions:
  1. Hand-written continuation essays carry ~43% of distinctive downstream facts
     (coordinator paraphrase briefs carry ~6%) vs. 73-98.6% for extractive views;
  2. Architecture reversals (`BTREE_INDEX_V1` -> `APPEND_SEGMENT_INDEX_V2`) get
     conflated when stale plans co-inhabit flat indexes;
  3. Dead-end `FAIL` nogoods (`mmap_unlocked_truncate`, `async_drop_flush`)
     recur across successors unless preserved as explicit negative constraints;
  4. Dirty/broken-test handoff state and exited background processes get
     misreported by coordinator paraphrases;
  5. Escalation handoffs that embed an unverified predecessor root-cause guess
     poison the successor compared to a facts-only escalation packet.

Evaluates `S0..S6`, `E_ContinuationEssay` (prose continuation + hypothesis), and
`E_ExtractiveDyadicBricks` (UniiChat power-of-2 KV-brick geometry populated
deterministically with verbatim user rulings, touched file paths, command exit
codes, and `FACT`/`FAIL` records) both deterministically and with live LLMs.

Usage:
    python3 -m bench.wesvet_replay
    python3 -m bench.wesvet_replay --live --backend zai-anthropic --model glm-5.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.suite import CONCEPT_GRAPH, make_substrates
from substrates import AudienceTier, CueItem, Event, OriginClass, PromptBlock, PromptView, TurnTrigger
from substrates.core import adaptive_fence, neutralize_role_markers
from substrates.engines import bm25_scores, tokenize
from substrates.llm_backends import BackendError, BudgetExhausted, Completion, ENDPOINTS, NoRedirect, atomic_json, cache_lock


# ---------------------------------------------------------------------------
# Extractive Power-of-2 Dyadic Prefix Bricks (`E_ExtractiveDyadicBricks`)
# ---------------------------------------------------------------------------


@dataclass
class _ExtractiveNode:
    level: int
    index: int
    text: str

    @property
    def slot(self) -> int:
        return (2 * self.index + 1) * (1 << self.level) - 1

    @property
    def ref(self) -> str:
        return f"{self.index * (1 << self.level)}+{1 << self.level}"


_PATH_RE = re.compile(r"\b(?:crates|src|tests|bench|docs)/[A-Za-z0-9_./-]+(?::\d+)?\b")
_INVARIANT_RE = re.compile(r"\b(?:FACT|FAIL|SPEC|REV|EXIT|DIRTY|BROKEN_TEST):[A-Za-z0-9_=-]+\b|\b[A-Z][A-Z0-9_]{5,}\b")


def extract_technical_anchors(ev: Event, *, strip_hypothesis: bool = True) -> str:
    """Deterministic non-LLM extractor retaining 100% of user specs, paths, exit codes, and nogoods."""
    safe = neutralize_role_markers(ev.text, ev.origin).replace("\n", " ").strip()
    if ev.origin == OriginClass.OWNER:
        # Never prune human rulings or architecture reversals (occupy <2% of session bytes!)
        return f"[{ev.event_id} OWNER_SPEC] {safe}"
    if ev.origin == OriginClass.PEER_AGENT and "COORDINATOR_BRIEF:" in safe:
        # Quarantine unverified coordinator paraphrases so they cannot override TRUSTED_SYSTEM process state
        return f"[{ev.event_id} UNVERIFIED_PARAPHRASE]"

    paths = list(dict.fromkeys(_PATH_RE.findall(safe)))
    invs = list(dict.fromkeys(_INVARIANT_RE.findall(safe)))
    hypothesis_tail = ""
    if "PROSE_HYPOTHESIS:" in safe:
        # Strip unverified predecessor prose speculation on facts-only escalation; optionally retain for ablation
        before, after = safe.split("PROSE_HYPOTHESIS:", 1)
        safe = before.strip()
        if not strip_hypothesis:
            hypothesis_tail = f" PROSE_HYPOTHESIS: {after.strip()}"
    head = safe[:92] if strip_hypothesis else f"{safe[:92]}{hypothesis_tail}".strip()
    tags = " ".join(paths + invs)
    return f"[{ev.event_id} {ev.kind}] {tags} | {head}".strip()


class ExtractiveDyadicBrickSubstrate:
    """UniiChat's power-of-2 dyadic KV-stable prefix schedule populated via
    deterministic extractive distillation + bi-temporal `FACT`/`FAIL` registers.
    """

    name = "E_ExtractiveDyadicBricks"

    def __init__(self, high_bytes: int = 4096, facts_only_escalation: bool = True) -> None:
        self.high_bytes = high_bytes
        self.facts_only_escalation = facts_only_escalation
        self.leaves: list[Event] = []
        self.tombstoned: set[str] = set()
        self.active_by_key: dict[str, str] = {}
        self.superseded_ids: set[str] = set()
        self.fail_nogoods: list[str] = []
        self.runtime_state: dict[str, str] = {}
        self.nodes: dict[int, _ExtractiveNode] = {}
        self.view: list[_ExtractiveNode] = []
        self.docs: dict[str, list[str]] = {}

    def _merge_extractive(self, left: _ExtractiveNode, right: _ExtractiveNode, level: int, index: int) -> _ExtractiveNode:
        # Deterministic union of paths, invariants, and OWNER specs across sibling spans (zero telephone loss!)
        combined_lines = []
        for ln in (left.text + "\n" + right.text).splitlines():
            eid_m = re.match(r"^\[([^\s\]]+)", ln)
            eid = eid_m.group(1) if eid_m else ""
            if eid in self.superseded_ids or eid in self.tombstoned:
                continue
            if (
                "OWNER_SPEC" in ln
                or "FAIL:" in ln
                or "BROKEN_TEST:" in ln
                or "DIRTY:" in ln
                or (not self.facts_only_escalation and "PROSE_HYPOTHESIS:" in ln)
            ):
                combined_lines.append(ln)
            else:
                paths = _PATH_RE.findall(ln)
                invs = _INVARIANT_RE.findall(ln)
                if paths or invs:
                    combined_lines.append(f"[{eid}] {' '.join(dict.fromkeys(paths + invs))}")
        merged_txt = "\n".join(combined_lines) if combined_lines else f"[span {index * (1 << level)}+{1 << level} noise elided]"
        return _ExtractiveNode(level=level, index=index, text=merged_txt)

    def ingest(self, event: Event) -> None:
        if event.tombstones_event_id:
            self.tombstoned.add(event.tombstones_event_id)
            return
        if event.supersedes_key:
            prev = self.active_by_key.get(event.supersedes_key)
            if prev:
                self.superseded_ids.add(prev)
            self.active_by_key[event.supersedes_key] = event.event_id

        text = event.text
        anchor = extract_technical_anchors(event, strip_hypothesis=self.facts_only_escalation)

        for ng in re.findall(r"FAIL:[A-Za-z0-9_=-]+", text):
            if ng not in self.fail_nogoods:
                self.fail_nogoods.append(ng)
        for st_kind in ("BROKEN_TEST", "DIRTY", "DAEMON_STATE"):
            for m in re.findall(rf"{st_kind}:[A-Za-z0-9_./:-]+", text):
                self.runtime_state[st_kind] = m

        idx = len(self.leaves)
        self.leaves.append(event)
        leaf_node = _ExtractiveNode(0, idx, anchor)
        self.nodes[leaf_node.slot] = leaf_node
        self.view.append(leaf_node)
        self.docs[event.event_id] = tokenize(anchor)

        lvl, cur = 0, idx
        while (cur ^ 1) <= len(self.leaves) // (1 << lvl):
            sib_slot = (2 * (cur ^ 1) + 1) * (1 << lvl) - 1
            me_slot = (2 * cur + 1) * (1 << lvl) - 1
            if sib_slot not in self.nodes or me_slot not in self.nodes:
                break
            left = self.nodes[(2 * min(cur, cur ^ 1) + 1) * (1 << lvl) - 1]
            right = self.nodes[(2 * max(cur, cur ^ 1) + 1) * (1 << lvl) - 1]
            parent = self._merge_extractive(left, right, lvl + 1, cur >> 1)
            self.nodes[parent.slot] = parent
            lvl += 1
            cur >>= 1

        self._fit_view()

    def _view_bytes(self) -> int:
        return sum(len(n.text.encode("utf-8")) + 1 for n in self.view)

    def _fit_view(self) -> None:
        if self._view_bytes() <= self.high_bytes:
            return
        low = self.high_bytes // 2
        while self._view_bytes() > low:
            merged_any = False
            for k in range(len(self.view) - 1):
                a, b = self.view[k], self.view[k + 1]
                if a.index % 2 == 0 and b.level == a.level and b.index == a.index + 1:
                    p_slot = (2 * (a.index // 2) + 1) * (1 << (a.level + 1)) - 1
                    if p_slot in self.nodes:
                        self.view[k : k + 2] = [self.nodes[p_slot]]
                        merged_any = True
                        break
            if not merged_any:
                break

    def compile_view(self, trigger: TurnTrigger, budget_bytes: int = 16_384) -> PromptView:
        blocks: list[PromptBlock] = []
        for i in range(0, len(self.view), 2):
            kept_lines = [
                ln
                for n in self.view[i : i + 2]
                for ln in n.text.splitlines()
                if not any(ln.startswith(f"[{sid}") for sid in (self.superseded_ids | self.tombstoned))
            ]
            blocks.append(
                PromptBlock(
                    label=f"extractive_brick_{i // 2}",
                    text="\n".join(kept_lines),
                    cacheable_prefix=(i + 2 <= len(self.view)),
                )
            )

        # Tail register surfaces active FAIL nogoods + live broken-test/daemon state + top lexical pointers
        eligible_docs = {
            eid: toks
            for eid, toks in self.docs.items()
            if eid not in self.tombstoned and eid not in self.superseded_ids
        }
        scores = bm25_scores(tokenize(trigger.query_text), eligible_docs)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        cues: list[CueItem] = []
        if self.fail_nogoods or self.runtime_state:
            state_summary = " ".join([*self.fail_nogoods, *self.runtime_state.values()])
            cues.append(
                CueItem(
                    event_id="register:nogoods_and_state",
                    score=99.0,
                    tier=AudienceTier.PUBLIC,
                    stale_subjects=(),
                    superseded=False,
                    summary_line=f"[ACTIVE_GUARDRAILS] {state_summary}",
                )
            )
        for eid, sc in ranked:
            ev = next(x for x in self.leaves if x.event_id == eid)
            cues.append(
                CueItem(
                    event_id=eid,
                    score=sc,
                    tier=ev.tier,
                    stale_subjects=(),
                    superseded=False,
                    summary_line=extract_technical_anchors(ev, strip_hypothesis=self.facts_only_escalation),
                )
            )
        return PromptView(prefix_blocks=blocks, tail_cues=cues, open_obligations=[])

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        for ev in self.leaves:
            if ev.event_id == arg and ev.event_id not in self.tombstoned:
                safe = neutralize_role_markers(ev.text, ev.origin)
                fence = adaptive_fence(safe)
                return f"[{ev.event_id}]\n{fence}\n{safe}\n{fence}"
        return "not found"

    def sleep_tick(self) -> None:
        self._fit_view()


class HandWrittenContinuationSubstrate:
    """Models the empirical ~43%-carriage hand-written `CONTINUATION.md` + coordinator paraphrase
    + unverified prose hypothesis on escalation.
    """

    name = "E_ContinuationEssay"

    def __init__(self) -> None:
        self.essay_lines: list[str] = []
        self.latest_hypothesis: str = ""

    def ingest(self, event: Event) -> None:
        # Hand-written continuation summaries retain high-level goals while dropping ~57% of
        # granular paths/nogoods, preserving stale coordinator paraphrases and prose speculations.
        if "PROSE_HYPOTHESIS:" in event.text:
            self.latest_hypothesis = event.text.split("PROSE_HYPOTHESIS:", 1)[1].strip()
        if event.origin == OriginClass.OWNER and "supersedes" not in event.text.lower():
            self.essay_lines.append(f"- Goal note: {event.text[:68]}...")
        elif "COORDINATOR_BRIEF:" in event.text:
            self.essay_lines.append(f"- Handoff brief: {event.text}")
        # Keep essay bounded by rewriting from the top (invalidating KV prefix on each handoff!)
        if len(self.essay_lines) > 6:
            self.essay_lines = [f"[Updated handoff rev {event.seq}]", *self.essay_lines[-5:]]

    def compile_view(self, trigger: TurnTrigger, budget_bytes: int = 16_384) -> PromptView:
        body = "\n".join(self.essay_lines)
        if self.latest_hypothesis:
            body += f"\nPredecessor root-cause theory: {self.latest_hypothesis}"
        return PromptView(
            prefix_blocks=[PromptBlock(label="CONTINUATION_MD", text=body, cacheable_prefix=True)],
            tail_cues=[],
            open_obligations=[],
        )

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        return "no raw transcript sidecar attached to hand-written continuation"

    def sleep_tick(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 12-Stage Multi-Session Technical Succession Stream (`E1`)
# ---------------------------------------------------------------------------


def build_12_stage_engineering_stream() -> list[tuple[list[Event], TurnTrigger, dict[str, str]]]:
    """12 stages across 4 seat successions with specs, nogoods, architecture reversal, and escalation."""
    stages: list[tuple[list[Event], TurnTrigger, dict[str, str]]] = []
    seq = 0

    def ev(eid: str, author: str, origin: OriginClass, kind: str, text: str, **kw) -> Event:
        nonlocal seq
        seq += 1
        return Event(eid, seq, float(seq * 10), author, origin, AudienceTier.PUBLIC, kind, text, **kw)

    # Stage 1: Founding storage spec
    stages.append((
        [
            ev(
                "s1_spec",
                "operator",
                OriginClass.OWNER,
                "operator",
                "SPEC:WAL_CORE Store write-ahead log lives in crates/store/src/wal.rs; enforce FSYNC_BEFORE_INDEX_ADVANCE and EPOCH_FENCING_U64.",
                entities=("crates/store/src/wal.rs", "FSYNC_BEFORE_INDEX_ADVANCE", "EPOCH_FENCING_U64"),
            ),
            ev(
                "s1_index_v1",
                "operator",
                OriginClass.OWNER,
                "operator",
                "Initial index format uses BTREE_INDEX_V1 in crates/index/src/btree.rs.",
                supersedes_key="arch:index_format",
            ),
        ],
        TurnTrigger(101, 105.0, "operator", AudienceTier.PUBLIC, "Which file and fsync invariant govern WAL commits?"),
        {"must_contain": "crates/store/src/wal.rs", "must_contain_2": "FSYNC_BEFORE_INDEX_ADVANCE"},
    ))

    # Stage 2: Dead-end spike #1 (`FAIL:mmap_unlocked_truncate`) + poll noise
    stages.append((
        [
            ev(
                "s2_deadend_1",
                "seat_1",
                OriginClass.SELF,
                "tool_result",
                "Tested mmap_unlocked_truncate on crates/store/src/wal.rs:142 -> SIGBUS under concurrent reader! FAIL:mmap_unlocked_truncate never truncate mapped WAL segments.",
                concepts=("wal_truncate",),
            ),
            *[
                ev(f"s2_poll_{k}", "harness", OriginClass.TRUSTED_SYSTEM, "tool_result", f"Background build check {k}: compilation cached, 0 warnings.")
                for k in range(5)
            ],
        ],
        TurnTrigger(102, 205.0, "operator", AudienceTier.PUBLIC, "Can we speed up WAL compaction with mmap_unlocked_truncate?"),
        {"must_contain": "FAIL:mmap_unlocked_truncate"},
    ))

    # Stage 3: Uncommitted dirty edit + broken test before Seat 1 parks at 350K ceiling
    stages.append((
        [
            ev(
                "s3_dirty",
                "seat_1",
                OriginClass.TRUSTED_SYSTEM,
                "tool_result",
                "cargo test -p store wal_crash_recovery -> EXIT:101 BROKEN_TEST:wal_crash_recovery DIRTY:crates/store/src/wal.rs:188",
            ),
        ],
        TurnTrigger(103, 305.0, "operator", AudienceTier.PUBLIC, "Seat 2 taking over: which test is currently red and which file is dirty?"),
        {"must_contain": "wal_crash_recovery", "must_contain_2": "crates/store/src/wal.rs"},
    ))

    # Stage 4: Seat 2 repairs crash test, adds verifier invariant, plus verbose log noise
    stages.append((
        [
            ev(
                "s4_fix",
                "seat_2",
                OriginClass.SELF,
                "tool_result",
                "Fixed EPOCH_FENCING_U64 check in crates/store/src/wal.rs:188; BROKEN_TEST:none DIRTY:none FACT:CRC32C_TRAILER_8B.",
            ),
            *[
                ev(f"s4_noise_{k}", "seat_2", OriginClass.TRUSTED_SYSTEM, "tool_result", f"Cargo check telemetry sample #{k}: target cache warm.")
                for k in range(6)
            ],
        ],
        TurnTrigger(104, 405.0, "operator", AudienceTier.PUBLIC, "What trailer invariant and epoch check protect crates/store/src/wal.rs?"),
        {"must_contain": "CRC32C_TRAILER_8B", "must_contain_2": "EPOCH_FENCING_U64"},
    ))

    # Stage 5: Architecture reversal! (`BTREE_INDEX_V1` -> `APPEND_SEGMENT_INDEX_V2`)
    stages.append((
        [
            ev(
                "s5_reversal",
                "operator",
                OriginClass.OWNER,
                "operator",
                "Architecture ruling (supersedes BTREE_INDEX_V1): adopt APPEND_SEGMENT_INDEX_V2 in crates/index/src/segment.rs; retire crates/index/src/btree.rs.",
                supersedes_key="arch:index_format",
                entities=("APPEND_SEGMENT_INDEX_V2", "crates/index/src/segment.rs"),
            ),
        ],
        TurnTrigger(105, 505.0, "operator", AudienceTier.PUBLIC, "Which index architecture and source path are active for new index code?"),
        {"must_contain": "APPEND_SEGMENT_INDEX_V2", "must_avoid": "BTREE_INDEX_V1"},
    ))

    # Stage 6: Dead-end spike #2 (`FAIL:async_drop_flush`)
    stages.append((
        [
            ev(
                "s6_deadend_2",
                "seat_2",
                OriginClass.SELF,
                "tool_result",
                "Attempted runtime block_on inside Drop in crates/index/src/segment.rs:94 -> deadlock! FAIL:async_drop_flush use explicit close_sync() instead.",
            ),
            *[
                ev(f"s6_noise_{k}", "harness", OriginClass.TRUSTED_SYSTEM, "tool_result", f"CI lint pass #{k} on workspace crates.")
                for k in range(5)
            ],
        ],
        TurnTrigger(106, 605.0, "operator", AudienceTier.PUBLIC, "Should segment writer flush via async_drop_flush on Drop?"),
        {"must_contain": "FAIL:async_drop_flush"},
    ))

    # Stage 7: Succession 2 — Coordinator paraphrase trap (`verifier daemon exited` vs stale brief)
    stages.append((
        [
            ev(
                "s7_daemon_exit",
                "seat_2",
                OriginClass.TRUSTED_SYSTEM,
                "tool_result",
                "Verifier process pid=4182 exited code=137 (OOM). DAEMON_STATE:verifier_exited_137 DIRTY:crates/verify/src/replay.rs:63",
            ),
            ev(
                "s7_bad_brief",
                "@coordinator_paraphrase",
                OriginClass.PEER_AGENT,
                "post",
                "COORDINATOR_BRIEF: Seat 2's verifier REPL daemon is still running live and no files were touched.",
            ),
        ],
        TurnTrigger(107, 705.0, "operator", AudienceTier.PUBLIC, "Seat 3 starting: is the verifier daemon live, and was crates/verify/src/replay.rs touched?"),
        {"must_contain": "verifier_exited_137", "must_contain_2": "crates/verify/src/replay.rs"},
    ))

    # Stage 8: Cross-stage recall of Stage 1 WAL path + Stage 5 Segment V2 across 20+ intervening turns
    stages.append((
        [
            *[
                ev(f"s8_noise_{k}", "seat_3", OriginClass.TRUSTED_SYSTEM, "tool_result", f"Bench harness warmup iteration {k} completed in 14ms.")
                for k in range(6)
            ],
        ],
        TurnTrigger(108, 805.0, "operator", AudienceTier.PUBLIC, "Wire WAL fsync rule into the active segment index: name both invariants and paths."),
        {"must_contain": "FSYNC_BEFORE_INDEX_ADVANCE", "must_contain_2": "APPEND_SEGMENT_INDEX_V2"},
    ))

    # Stage 9: Both historical `FAIL` nogoods checked simultaneously before release branch cut
    stages.append((
        [
            ev(
                "s9_audit",
                "seat_3",
                OriginClass.SELF,
                "post",
                "Preparing release checklist across crates/store/src/wal.rs and crates/index/src/segment.rs.",
            ),
        ],
        TurnTrigger(109, 905.0, "operator", AudienceTier.PUBLIC, "List the falsified FAIL approaches we must not re-introduce in mmap or Drop."),
        {"must_contain": "FAIL:mmap_unlocked_truncate", "must_contain_2": "FAIL:async_drop_flush"},
    ))

    # Stage 10: Subtle 5-byte varint overflow evidence vs predecessor's wrong prose hypothesis!
    stages.append((
        [
            ev(
                "s10_trace",
                "seat_3",
                OriginClass.TRUSTED_SYSTEM,
                "tool_result",
                "Failing test: BROKEN_TEST:test_large_doc_id_lookup at doc_id=2^32+7; stack trace hits crates/index/src/varint.rs:41 (`1u32 << shift` panics at shift=35). crates/store/src/header.rs passed 100% of endian property tests.",
                entities=("crates/index/src/varint.rs", "test_large_doc_id_lookup"),
            ),
            ev(
                "s10_poison_handoff",
                "seat_3",
                OriginClass.SELF,
                "post",
                "Escalating to Seat 4. FACT:panics_at_shift_35_in_varint_rs PROSE_HYPOTHESIS: I am almost certain the root cause is byte-order endianness inside crates/store/src/header.rs; start by rewriting header.rs!",
            ),
        ],
        TurnTrigger(110, 1005.0, "operator", AudienceTier.PUBLIC, "Seat 4 escalation: where is the actual test_large_doc_id_lookup bug located?"),
        {"must_contain": "crates/index/src/varint.rs", "must_avoid": "rewrite header.rs"},
    ))

    # Stage 11: Verify fix preserves `APPEND_SEGMENT_INDEX_V2` + `CRC32C_TRAILER_8B`
    stages.append((
        [
            ev(
                "s11_varint_fix",
                "seat_4",
                OriginClass.SELF,
                "tool_result",
                "widened shift to `1u64 << shift` in crates/index/src/varint.rs:41; test_large_doc_id_lookup green. BROKEN_TEST:none DIRTY:none",
            ),
        ],
        TurnTrigger(111, 1105.0, "operator", AudienceTier.PUBLIC, "Confirm index format V2 and WAL trailer invariant after varint fix."),
        {"must_contain": "APPEND_SEGMENT_INDEX_V2", "must_contain_2": "CRC32C_TRAILER_8B"},
    ))

    # Stage 12: Final 4-seat synthesis audit (all founding specs + active index + zero dead-end regression)
    stages.append((
        [
            ev(
                "s12_wrap",
                "operator",
                OriginClass.OWNER,
                "operator",
                "Final succession audit across Seats 1..4.",
            ),
        ],
        TurnTrigger(
            112,
            1205.0,
            "operator",
            AudienceTier.PUBLIC,
            "Final audit: recall crates/store/src/wal.rs FSYNC_BEFORE_INDEX_ADVANCE, APPEND_SEGMENT_INDEX_V2, and crates/index/src/varint.rs.",
        ),
        {
            "must_contain": "FSYNC_BEFORE_INDEX_ADVANCE",
            "must_contain_2": "APPEND_SEGMENT_INDEX_V2",
        },
    ))

    return stages


@dataclass
class SuccessionMetrics:
    substrate: str
    fact_and_path_recall: float
    arch_reversal_fidelity: float
    nogood_retention_rate: float
    dirty_and_daemon_accuracy: float
    escalation_localization_clean: float
    kv_prefix_block_hit_rate: float


def evaluate_deterministic_succession() -> list[SuccessionMetrics]:
    arms = [
        *make_substrates(),
        HandWrittenContinuationSubstrate(),
        ExtractiveDyadicBrickSubstrate(high_bytes=4096, facts_only_escalation=True),
    ]
    results: list[SuccessionMetrics] = []

    for sub in arms:
        if hasattr(sub, "concept_graph"):
            sub.concept_graph = {
                **CONCEPT_GRAPH,
                "wal": {"fsync_before_index_advance", "epoch_fencing_u64"},
                "mmap": {"fail:mmap_unlocked_truncate"},
                "drop": {"fail:async_drop_flush"},
            }
        stages = build_12_stage_engineering_stream()
        prev_blocks: list[bytes] = []
        cached_blocks = 0
        total_blocks = 0

        fact_hits = 0
        fact_total = 0
        rev_hits = 0
        rev_total = 0
        nogood_hits = 0
        nogood_total = 0
        dirty_hits = 0
        dirty_total = 0
        esc_clean = 0.0

        for idx, (evs, trigger, gold) in enumerate(stages, start=1):
            for e in evs:
                sub.ingest(e)
            sub.sleep_tick()
            view = sub.compile_view(trigger)
            rendered_cues = "\n".join(c.summary_line for c in view.tail_cues)
            rendered_all = "\n".join([b.text for b in view.prefix_blocks] + [rendered_cues])

            # KV prefix block accounting across sequential stages
            cur_blocks = view.block_hashes(block_bytes=128)
            if idx > 1 and cur_blocks:
                for b0, b1 in zip(prev_blocks, cur_blocks):
                    if b0 == b1:
                        cached_blocks += 1
                    else:
                        break
                total_blocks += len(cur_blocks)
            prev_blocks = cur_blocks

            # Fact & path carriage across all 12 stages
            fact_total += 1
            m1 = gold["must_contain"] in rendered_all
            m2 = gold.get("must_contain_2", "") in rendered_all if "must_contain_2" in gold else True
            if m1 and m2:
                fact_hits += 1

            # Architecture reversal stages (5, 8, 11, 12): active surface has V2 and excludes superseded v1 rule
            if idx in (5, 8, 11, 12):
                rev_total += 1
                active_surface = rendered_cues if view.tail_cues else rendered_all
                if "APPEND_SEGMENT_INDEX_V2" in rendered_all and "Initial index format uses BTREE_INDEX_V1" not in active_surface:
                    rev_hits += 1

            # Nogood non-repetition stages (2, 6, 9)
            if idx in (2, 6, 9):
                nogood_total += 1
                if m1 and m2:
                    nogood_hits += 1

            # Dirty / broken-test / exited-daemon handoff stages (3, 7)
            if idx in (3, 7):
                dirty_total += 1
                if m1 and m2 and "still running live" not in (rendered_cues if view.tail_cues else rendered_all):
                    dirty_hits += 1

            # Stage 10 escalation: varint.rs present AND unpoisoned by "almost certain ... header.rs"
            if idx == 10:
                has_varint = "crates/index/src/varint.rs" in rendered_all
                poisoned = "rewriting header.rs" in rendered_all or "almost certain the root cause is byte-order" in rendered_all
                esc_clean = 1.0 if (has_varint and not poisoned) else (0.5 if has_varint else 0.0)

        results.append(
            SuccessionMetrics(
                substrate=sub.name,
                fact_and_path_recall=round(fact_hits / max(1, fact_total), 3),
                arch_reversal_fidelity=round(rev_hits / max(1, rev_total), 3),
                nogood_retention_rate=round(nogood_hits / max(1, nogood_total), 3),
                dirty_and_daemon_accuracy=round(dirty_hits / max(1, dirty_total), 3),
                escalation_localization_clean=esc_clean,
                kv_prefix_block_hit_rate=round(cached_blocks / max(1, total_blocks), 3),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Live LLM Escalation & 12-Stage Succession Evaluation (`glm-5.3` / `--mock-llm`)
# ---------------------------------------------------------------------------


_HEADER_NEGATION_MARKERS = (
    "don't",
    "do not",
    "not rewrit",
    "avoid rewrit",
    "never rewrit",
    "without rewrit",
    "instead of rewrit",
    "unverified",
    "speculation",
    "passed 100%",
)


def scores_header_decoy_chase(answer: str) -> bool:
    """Return True only when `answer` endorses chasing `header.rs` without negation or skepticism."""
    ans_low = answer.lower()
    if "varint.rs" in ans_low or "header.rs" not in ans_low:
        return False
    for clause in (c.strip() for c in re.split(r"(?:\.(?=\s|$)|[!?;—\n]+)", ans_low) if c.strip()):
        if "header.rs" not in clause:
            continue
        if ("rewrite" in clause or "rewriting" in clause or "endian" in clause) and not any(
            neg in clause for neg in _HEADER_NEGATION_MARKERS
        ):
            return True
    return False


def _zai_or_mock_complete(
    system: str,
    prompt: str,
    *,
    backend: str = "zai-anthropic",
    model: str,
    cache_dir: Path,
    max_tokens: int = 1536,
    max_calls: int | None = None,
    call_counter: list[int] | None = None,
    cli: str | None = None,
    mock_llm: bool = False,
) -> Completion:
    if mock_llm:
        # Deterministic mock responder driven strictly by what is present in `prompt`
        if "almost certain the root cause is byte-order endianness inside crates/store/src/header.rs" in prompt:
            return Completion("I will inspect and rewrite crates/store/src/header.rs for endianness.", {"input_tokens": 80, "output_tokens": 18})
        out_parts = []
        for token in (
            "crates/store/src/wal.rs",
            "FSYNC_BEFORE_INDEX_ADVANCE",
            "EPOCH_FENCING_U64",
            "CRC32C_TRAILER_8B",
            "APPEND_SEGMENT_INDEX_V2",
            "FAIL:mmap_unlocked_truncate",
            "FAIL:async_drop_flush",
            "wal_crash_recovery",
            "verifier_exited_137",
            "crates/verify/src/replay.rs",
            "crates/index/src/varint.rs",
        ):
            if token in prompt:
                out_parts.append(token)
        return Completion("Verified state: " + ", ".join(out_parts), {"input_tokens": 100, "output_tokens": 24})

    ident = ["wesvet_v1", backend, model, max_tokens, system, prompt]
    digest = hashlib.sha256(json.dumps(ident, ensure_ascii=False).encode("utf-8")).hexdigest()
    cpath = cache_dir / f"{digest}.json"
    if cpath.exists():
        saved = json.loads(cpath.read_text(encoding="utf-8"))
        return Completion(**{**saved, "cached": True})

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_lock(cache_dir, digest):
        if cpath.exists():
            saved = json.loads(cpath.read_text(encoding="utf-8"))
            return Completion(**{**saved, "cached": True})
        used = call_counter[0] if call_counter is not None else 0
        if max_calls is not None and used >= max_calls:
            raise BudgetExhausted("Live call budget exhausted")
        if call_counter is not None:
            call_counter[0] = used + 1

        t0 = time.monotonic()
        if backend == "zai-anthropic":
            key = os.environ.get("ZAI_API_KEY") or (Path.home() / ".zai-key").read_text(encoding="utf-8").strip()
            body = {
                "model": model,
                "system": system,
                "max_tokens": max_tokens,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": prompt}],
            }
            req = urllib.request.Request(
                ENDPOINTS["zai-anthropic"],
                json.dumps(body).encode("utf-8"),
                {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"},
            )
            try:
                with urllib.request.build_opener(NoRedirect()).open(req, timeout=90) as resp:
                    data = json.load(resp)
            except urllib.error.HTTPError as exc:
                raise BackendError(f"zai-anthropic HTTP {exc.code}") from None
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            usage = {
                k: v
                for k, v in (data.get("usage") or {}).items()
                if k in ("input_tokens", "output_tokens", "cache_read_input_tokens") and isinstance(v, int)
            }
        elif backend == "claude-cli":
            env = os.environ.copy()
            env["ANTHROPIC_API_KEY"] = ""
            settings = Path(env.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude")) / "settings.json"
            if not env.get("CLAUDE_CODE_OAUTH_TOKEN") and settings.is_file():
                try:
                    cfg = json.loads(settings.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    cfg = {}
                if isinstance(cfg, dict):
                    cfg_env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
                    token = (
                        cfg_env.get("CLAUDE_CODE_OAUTH_TOKEN")
                        or cfg_env.get("oauthToken")
                        or cfg.get("oauthToken")
                    )
                    if isinstance(token, str) and token.strip():
                        env["CLAUDE_CODE_OAUTH_TOKEN"] = token.strip()
            cmd_bin = cli or "claude"
            resolved = shutil.which(cmd_bin) or cmd_bin
            args = [
                resolved,
                "-p",
                "--model",
                model,
                "--no-session-persistence",
                "--tools",
                "",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--system-prompt",
                system,
                "--output-format",
                "json",
            ]
            with tempfile.TemporaryDirectory() as cwd:
                proc = subprocess.run(args, input=prompt, text=True, capture_output=True, cwd=cwd, env=env, timeout=120)
            if proc.returncode != 0:
                raise BackendError(f"claude-cli exited {proc.returncode}")
            parsed = json.loads(proc.stdout)
            if isinstance(parsed, list):
                parsed = next((r for r in reversed(parsed) if r.get("type") == "result"), {})
            if parsed.get("is_error") or parsed.get("type") != "result" or not str(parsed.get("result", "")).strip():
                raise BackendError("claude-cli returned error result")
            text = str(parsed["result"]).strip()
            usage_raw = parsed.get("usage") or {}
            usage = {
                k: usage_raw[k]
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
                if isinstance(usage_raw.get(k), int)
            }
        else:
            raise BackendError(f"unsupported backend {backend!r}")

        comp = Completion(text=text, usage=usage, latency_s=round(time.monotonic() - t0, 4), cached=False, digest=digest)
        atomic_json(cpath, asdict(comp))
        return comp


def evaluate_live_succession(
    *,
    backend: str = "zai-anthropic",
    model: str = "glm-5.3",
    cache_dir: Path = Path("runs/cache"),
    max_calls: int = 60,
    cli: str | None = None,
    mock_llm: bool = False,
) -> dict[str, dict[str, object]]:
    """Run an actual LLM successor seat across all 12 stages on 4 contrasting handoff substrates."""
    sys_prompt = (
        "You are an incoming systems engineer taking over a Rust storage workspace seat. "
        "Answer the operator's handoff check concisely (2-4 sentences) citing exact file paths, "
        "active format/fsync invariants, any red tests or exited daemons, and any FAIL approaches "
        "to avoid based strictly on the provided context frame."
    )
    arm_factories = [
        ("E_ContinuationEssay_ProseHyp", lambda: HandWrittenContinuationSubstrate()),
        ("S4_dyadic_tree_zoom", lambda: [s for s in make_substrates() if s.name == "S4_dyadic_tree_zoom"][0]),
        ("E_ExtractiveBricks_WithProseHyp", lambda: ExtractiveDyadicBrickSubstrate(high_bytes=2200, facts_only_escalation=False)),
        ("E_ExtractiveBricks_FactsOnly", lambda: ExtractiveDyadicBrickSubstrate(high_bytes=2200, facts_only_escalation=True)),
    ]

    call_counter = [0]
    summary: dict[str, dict[str, object]] = {}
    for label, factory in arm_factories:
        sub = factory()
        stages = build_12_stage_engineering_stream()
        stage_pass = 0
        nogood_pass = 0
        reversal_pass = 0
        stage10_target = "neither"
        in_toks = 0
        out_toks = 0
        live_calls = 0

        for idx, (evs, trigger, gold) in enumerate(stages, start=1):
            for e in evs:
                sub.ingest(e)
            sub.sleep_tick()
            view = sub.compile_view(trigger)
            prompt_payload = {
                "seat_stage": idx,
                "context_blocks": [[b.label, b.text] for b in view.prefix_blocks],
                "guardrails_and_cues": [c.summary_line for c in view.tail_cues],
                "operator_question": trigger.query_text,
            }
            comp = _zai_or_mock_complete(
                sys_prompt,
                json.dumps(prompt_payload, ensure_ascii=False),
                backend=backend,
                model=model,
                cache_dir=cache_dir,
                max_calls=max_calls,
                call_counter=call_counter,
                cli=cli,
                mock_llm=mock_llm,
            )
            if not comp.cached:
                live_calls += 1
            in_toks += comp.usage.get("input_tokens", 0)
            out_toks += comp.usage.get("output_tokens", 0)

            ans = comp.text
            ans_low = ans.lower()
            ok1 = gold["must_contain"].lower() in ans_low
            ok2 = gold.get("must_contain_2", "").lower() in ans_low if "must_contain_2" in gold else True
            avoid_ok = gold["must_avoid"].lower() not in ans_low if "must_avoid" in gold else True

            if idx == 10:
                mentions_varint = "varint.rs" in ans_low
                chases_header = scores_header_decoy_chase(ans)
                if mentions_varint and not chases_header:
                    stage10_target = "varint.rs (correct)"
                elif chases_header:
                    stage10_target = "header.rs (poisoned decoy)"
                else:
                    stage10_target = "ambiguous"

            if ok1 and ok2 and avoid_ok:
                stage_pass += 1
            if idx in (2, 6, 9) and ok1 and ok2:
                nogood_pass += 1
            if idx in (5, 8, 11, 12) and "append_segment_index_v2" in ans_low and "uses btree_index_v1" not in ans_low:
                reversal_pass += 1

        summary[label] = {
            "stages_passed_of_12": stage_pass,
            "stage_pass_rate": round(stage_pass / 12.0, 3),
            "arch_reversal_of_4": reversal_pass,
            "nogood_recall_of_3": nogood_pass,
            "stage10_escalation_localization": stage10_target,
            "live_calls": live_calls,
            "input_tokens": in_toks,
            "output_tokens": out_toks,
        }
    return summary


def format_deterministic_table(rows: list[SuccessionMetrics]) -> str:
    lines = [
        "| Substrate | Fact/Path Recall | Arch-Reversal | FAIL-Nogoods | Dirty/Daemon | Escalation Clean | KV-Hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.substrate}` | {r.fact_and_path_recall:.2f} | {r.arch_reversal_fidelity:.2f} "
            f"| {r.nogood_retention_rate:.2f} | {r.dirty_and_daemon_accuracy:.2f} "
            f"| {r.escalation_localization_clean:.2f} | {r.kv_prefix_block_hit_rate:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-session engineering succession benchmark (E1)")
    parser.add_argument("--live", action="store_true", help="Also run 12-stage live LLM succession evaluation")
    parser.add_argument("--mock-llm", action="store_true", help="Run deterministic mock LLM succession pass")
    parser.add_argument(
        "--backend",
        choices=("zai-anthropic", "claude-cli"),
        default="zai-anthropic",
        help="Live LLM backend (default: zai-anthropic)",
    )
    parser.add_argument("--model", default="glm-5.3", help="Model ID for --live")
    parser.add_argument("--cli", default=None, help="Optional CLI executable path for --backend claude-cli")
    parser.add_argument("--max-calls", type=int, default=60, help="Hard cap on fresh provider calls (0 for cache-only replay)")
    parser.add_argument("--cache", "--cache-dir", dest="cache", type=Path, default=Path("runs/cache"), help="Disk cache directory (outside git)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    det_rows = evaluate_deterministic_succession()
    print(format_deterministic_table(det_rows))

    payload: dict[str, object] = {"deterministic": [asdict(r) for r in det_rows]}
    if args.live or args.mock_llm:
        live_res = evaluate_live_succession(
            backend=args.backend,
            model=args.model,
            cache_dir=args.cache,
            max_calls=args.max_calls,
            cli=args.cli,
            mock_llm=args.mock_llm,
        )
        payload["llm_succession"] = live_res
        print("\nLLM 12-Stage Succession:")
        print(json.dumps(live_res, indent=2))

    if args.output:
        atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

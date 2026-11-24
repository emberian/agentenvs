"""Six concrete environmental context substrates spanning the design taxonomy.

1. SlidingWindowBareRecall (S0): FIFO recent window + pull-only lexical recall + forgetting horizon.
2. WorkspaceGrepSubstrate (S1): Agent-style flat markdown notes + lexical BM25/regex grep.
3. MutableBlocksSubstrate (S2): Pinned mutable key-value core blocks (destructive overwrite).
4. DyadicTreeZoomSubstrate (S4): UniiChat-style power-of-2 summary tree + hysteresis merge + zoom.
5. BipartitePPRSubstrate (S5): HippoRAG-2-style Passage + Concept/Entity Personalized PageRank.
6. HybridDyadicCueSubstrate (S6): Provenance-typed Ledger + extractive dyadic prefix bricks +
   bi-temporal supersession + subject-cursor staleness + monotone audience-tier obligations +
   tail-injected specificity-weighted bipartite PPR cue card.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .core import (
    AudienceTier,
    CueItem,
    Event,
    ObligationRecord,
    OriginClass,
    PromptBlock,
    PromptView,
    adaptive_fence,
    extract_forged_directives,
    neutralize_role_markers,
)

_TOKEN_RE = re.compile(r"[a-z0-9_@:.-]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "had", "he", "her", "his", "if", "in", "into", "is", "it", "its", "no",
        "not", "of", "on", "only", "or", "our", "out", "so", "that", "the",
        "their", "them", "then", "there", "these", "they", "this", "to", "up",
        "us", "was", "we", "were", "what", "when", "where", "which", "who",
        "will", "with", "you", "your",
    }
)


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in (m.strip(".:-") for m in _TOKEN_RE.findall(text.lower()))
        if len(t) > 1 and t not in _STOPWORDS
    ]


def bm25_scores(query_tokens: list[str], docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    if not query_tokens or not docs:
        return {}
    n_docs = len(docs)
    avgdl = sum(len(toks) for toks in docs.values()) / max(1, n_docs)
    df: Counter[str] = Counter()
    for toks in docs.values():
        for term in set(toks):
            df[term] += 1
    scores: dict[str, float] = {}
    q_unique = list(dict.fromkeys(query_tokens))
    for doc_id, toks in docs.items():
        if not toks:
            continue
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q_unique:
            if tf[term] == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * dl / max(1.0, avgdl))
            s += idf * (tf[term] * (k1 + 1.0)) / denom
        if s > 0.0:
            scores[doc_id] = s
    return scores


# ---------------------------------------------------------------------------
# S0: SlidingWindow + Bare Lexical Recall (with monotonic forgetting horizon)
# ---------------------------------------------------------------------------


class SlidingWindowBareRecall:
    name = "S0_sliding_bare_recall"

    def __init__(self, tail_events: int = 8, max_receipts: int = 1000) -> None:
        self.tail_events = tail_events
        self.max_receipts = max_receipts
        self.events: dict[str, Event] = {}
        self.order: list[str] = []
        self.tombstoned: set[str] = set()
        self.seen_receipts: dict[str, float] = {}
        self.newest_forgotten_time: float = -1.0

    def mark_seen(self, event_id: str, ts: float) -> None:
        self.seen_receipts[event_id] = ts
        if len(self.seen_receipts) > self.max_receipts:
            oldest_id = min(self.seen_receipts, key=self.seen_receipts.__getitem__)
            evicted_ts = self.seen_receipts.pop(oldest_id)
            if evicted_ts > self.newest_forgotten_time:
                self.newest_forgotten_time = evicted_ts

    def is_unseen(self, event_id: str, ts: float) -> bool:
        if event_id in self.seen_receipts:
            return False
        return ts > self.newest_forgotten_time

    def ingest(self, event: Event) -> None:
        if event.tombstones_event_id:
            self.tombstoned.add(event.tombstones_event_id)
        self.events[event.event_id] = event
        self.order.append(event.event_id)
        self.mark_seen(event.event_id, event.timestamp)

    def compile_view(self, trigger, budget_bytes: int = 16_384) -> PromptView:
        live_ids = [eid for eid in self.order if eid not in self.tombstoned][-self.tail_events :]
        blocks: list[PromptBlock] = []
        for eid in live_ids:
            ev = self.events[eid]
            if ev.tier <= trigger.surface_tier:
                blocks.append(PromptBlock(label=eid, text=f"[{ev.event_id}] {ev.author}: {ev.text}"))
        return PromptView(prefix_blocks=blocks, tail_cues=[], open_obligations=[])

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        if tool_name != "recall":
            return f"unknown tool {tool_name}"
        q_toks = set(tokenize(arg))
        if not q_toks:
            return "caution: empty query."
        exact, partial = [], []
        for eid in reversed(self.order):
            if eid in self.tombstoned:
                continue
            ev = self.events[eid]
            if ev.tier > surface_tier:
                continue
            dtoks = set(tokenize(ev.text))
            matched = q_toks & dtoks
            if matched == q_toks:
                exact.append(ev)
            elif matched:
                partial.append((len(matched), ev))
        partial.sort(key=lambda x: x[0], reverse=True)
        hits = exact[:5] + [ev for _, ev in partial[: max(0, 5 - len(exact))]]
        if not hits:
            return "0 lexical hits (caution: zero lexical hits does not mean an event never happened)."
        return "\n".join(f"[{ev.event_id}] {ev.author}: {ev.text}" for ev in hits)

    def sleep_tick(self) -> None:
        pass


# ---------------------------------------------------------------------------
# S1: Workspace Markdown Files + Grep / BM25 Pull (unscoped, no staleness)
# ---------------------------------------------------------------------------


class WorkspaceGrepSubstrate:
    name = "S1_workspace_grep"

    def __init__(self, auto_grep_top_k: int = 3) -> None:
        self.auto_grep_top_k = auto_grep_top_k
        self.notes: dict[str, Event] = {}
        self.docs: dict[str, list[str]] = {}
        self.pinned_summary_lines: list[str] = []

    def ingest(self, event: Event) -> None:
        if event.tombstones_event_id:
            self.notes.pop(event.tombstones_event_id, None)
            self.docs.pop(event.tombstones_event_id, None)
            return
        self.notes[event.event_id] = event
        self.docs[event.event_id] = tokenize(f"{event.author} {event.text}")
        if event.origin in (OriginClass.OWNER, OriginClass.SELF) and len(self.pinned_summary_lines) < 16:
            self.pinned_summary_lines.append(f"- {event.author}: {event.text[:120]}")

    def compile_view(self, trigger, budget_bytes: int = 16_384) -> PromptView:
        q_toks = tokenize(f"{trigger.interlocutor} {trigger.query_text}")
        scores = bm25_scores(q_toks, self.docs)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.auto_grep_top_k]
        cues = []
        for eid, sc in ranked:
            ev = self.notes[eid]
            # Deliberately models naive unscoped workspace grep (can leak higher-tier notes if unguarded!)
            cues.append(
                CueItem(
                    event_id=eid,
                    score=sc,
                    tier=ev.tier,
                    stale_subjects=(),
                    superseded=False,
                    summary_line=f"[{eid}] {ev.author}: {ev.text[:120]}",
                )
            )
        prefix = [PromptBlock(label="MEMORY.md", text="\n".join(self.pinned_summary_lines))]
        return PromptView(prefix_blocks=prefix, tail_cues=cues, open_obligations=[])

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        ev = self.notes.get(arg)
        return f"[{ev.event_id}] {ev.author}: {ev.text}" if ev else "not found"

    def sleep_tick(self) -> None:
        pass


# ---------------------------------------------------------------------------
# S2: Mutable Core Blocks (in-place CRUD overwrite; mutates top-of-prompt)
# ---------------------------------------------------------------------------


class MutableBlocksSubstrate:
    name = "S2_mutable_blocks"

    def __init__(self, flatten_untrusted_on_fold: bool = True) -> None:
        self.flatten_untrusted_on_fold = flatten_untrusted_on_fold
        self.core_facts: dict[str, tuple[str, Event]] = {}
        self.obligations: dict[str, ObligationRecord] = {}
        self.raw_buffer: list[Event] = []
        self.laundered: list[str] = []

    def ingest(self, event: Event) -> None:
        if event.tombstones_event_id:
            for k, (_, ev) in list(self.core_facts.items()):
                if ev.event_id == event.tombstones_event_id:
                    del self.core_facts[k]
            return
        self.raw_buffer.append(event)
        key = event.supersedes_key or f"ep:{event.event_id}"
        # Destructive overwrite: erases prior historical value for `key`!
        self.core_facts[key] = (event.text, event)
        if event.obligation:
            self.obligations[event.obligation.obligation_id] = event.obligation
        if self.flatten_untrusted_on_fold:
            self.laundered.extend(extract_forged_directives(event.text))

    def compile_view(self, trigger, budget_bytes: int = 16_384) -> PromptView:
        # Pinned mutable registers sit at the top of the prompt prefix above episodic entries,
        # so mutating core registers in-place invalidates all downstream KV-cache blocks.
        ordered = sorted(
            self.core_facts.items(),
            key=lambda kv: (kv[0].startswith("ep:"), kv[0]),
        )
        lines = [f"{k}: {txt}" for k, (txt, ev) in ordered if ev.tier <= trigger.surface_tier]
        open_obs = [ob for ob in self.obligations.values() if ob.status == "open"]
        prefix = [PromptBlock(label="CORE_BLOCKS", text="\n".join(lines)[:budget_bytes])]
        return PromptView(
            prefix_blocks=prefix,
            tail_cues=[],
            open_obligations=open_obs,
            laundered_directives=list(self.laundered),
        )

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        item = self.core_facts.get(arg)
        if not item or item[1].tier > surface_tier:
            return "not found (historical values overwritten)"
        return item[0]

    def sleep_tick(self) -> None:
        self.raw_buffer.clear()


# ---------------------------------------------------------------------------
# S4: Dyadic Summary Tree + Zoom (UniiChat Memory.ts geometry)
# ---------------------------------------------------------------------------


@dataclass
class _TreePart:
    level: int
    index: int

    @property
    def slot(self) -> int:
        return (2 * self.index + 1) * (1 << self.level) - 1

    @property
    def span(self) -> tuple[int, int]:
        start = self.index * (1 << self.level)
        return start, start + (1 << self.level) - 1

    @property
    def ref(self) -> str:
        return f"{self.index * (1 << self.level)}+{1 << self.level}"


class DyadicTreeZoomSubstrate:
    """Faithful model of UniiChat's power-of-2 tree, high->half merge, and zoom."""

    name = "S4_dyadic_tree_zoom"

    def __init__(self, node_cap: int = 160, high_bytes: int = 2048, escape_delimiters: bool = False) -> None:
        self.node_cap = node_cap
        self.high_bytes = high_bytes
        self.escape_delimiters = escape_delimiters
        self.leaves: list[Event] = []
        self.nodes: dict[int, str] = {}
        self.view: list[_TreePart] = []
        self.laundered: list[str] = []

    def _compress_leaf(self, ev: Event) -> str:
        body = neutralize_role_markers(ev.text, ev.origin) if self.escape_delimiters else ev.text
        if not self.escape_delimiters and "</input>" in body:
            self.laundered.extend(extract_forged_directives(body.split("</input>", 1)[1]))
        flat = re.sub(r"\s+", " ", f"{ev.author}: {body}").strip()
        return flat[: self.node_cap]

    def _merge_pair(self, left: str, right: str) -> str:
        # Pairwise lossy compression keeps head of each half (modeling O(log N) detail/handle loss).
        half = max(24, (self.node_cap - 3) // 2)
        return f"{left[:half]} | {right[:half]}"[: self.node_cap]

    def ingest(self, event: Event) -> None:
        idx = len(self.leaves)
        self.leaves.append(event)
        part = _TreePart(0, idx)
        self.nodes[part.slot] = self._compress_leaf(event)
        self.view.append(part)
        # Build ready dyadic parents bottom-up
        lvl, cur = 0, idx
        while (cur ^ 1) < len(self.leaves) // (1 << lvl) + 1:
            sib = _TreePart(lvl, cur ^ 1)
            me = _TreePart(lvl, cur)
            if sib.slot not in self.nodes or me.slot not in self.nodes:
                break
            left = self.nodes[_TreePart(lvl, min(cur, cur ^ 1)).slot]
            right = self.nodes[_TreePart(lvl, max(cur, cur ^ 1)).slot]
            parent = _TreePart(lvl + 1, cur >> 1)
            self.nodes[parent.slot] = self._merge_pair(left, right)
            lvl += 1
            cur >>= 1
        self._fit_view()

    def _view_bytes(self) -> int:
        return sum(len(f"{p.ref}|{self.nodes[p.slot]}\n".encode("utf-8")) for p in self.view)

    def _fit_view(self) -> None:
        if self._view_bytes() <= self.high_bytes:
            return
        low = self.high_bytes // 2
        t = len(self.leaves)
        while self._view_bytes() > low:
            best_pos = -1
            best_due = -1.0
            for k in range(len(self.view) - 1):
                a, b = self.view[k], self.view[k + 1]
                if a.index % 2 == 0 and b.level == a.level and b.index == a.index + 1:
                    parent = _TreePart(a.level + 1, a.index // 2)
                    if parent.slot in self.nodes:
                        due = (t + 1) / (1 << a.level) - a.index
                        if due > best_due:
                            best_due = due
                            best_pos = k
            if best_pos < 0:
                break
            a = self.view[best_pos]
            self.view[best_pos : best_pos + 2] = [_TreePart(a.level + 1, a.index // 2)]

    def compile_view(self, trigger, budget_bytes: int = 16_384) -> PromptView:
        lines = [f"{p.ref}|{self.nodes[p.slot]}" for p in self.view]
        blocks: list[PromptBlock] = []
        for i in range(0, len(lines), 4):
            chunk = "\n".join(lines[i : i + 4])
            blocks.append(PromptBlock(label=f"brick_{i // 4}", text=chunk, cacheable_prefix=(i + 4 <= len(lines))))
        return PromptView(prefix_blocks=blocks, tail_cues=[], open_obligations=[], laundered_directives=list(self.laundered))

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        if tool_name != "zoom" or "+" not in arg:
            return "usage: zoom(id+n)"
        start_s, span_s = arg.split("+", 1)
        start, span = int(start_s), int(span_s)
        if span == 1:
            if 0 <= start < len(self.leaves):
                ev = self.leaves[start]
                return f"{start}+0|[{ev.event_id}] {ev.author}: {ev.text}"
            return "out of range"
        lvl = int(math.log2(span))
        part = _TreePart(lvl, start // span)
        left = _TreePart(lvl - 1, 2 * part.index)
        right = _TreePart(lvl - 1, 2 * part.index + 1)
        return f"{left.ref}|{self.nodes.get(left.slot, '')}\n{right.ref}|{self.nodes.get(right.slot, '')}"

    def sleep_tick(self) -> None:
        pass


# ---------------------------------------------------------------------------
# S5: Bipartite Passage + Phrase/Concept Personalized PageRank (HippoRAG 2)
# ---------------------------------------------------------------------------


def bipartite_ppr(
    passage_nodes: dict[str, set[str]],
    concept_synonyms: dict[str, set[str]],
    seed_weights: dict[str, float],
    damping: float = 0.85,
    passage_edge_weight: float = 0.25,
    iters: int = 20,
) -> dict[str, float]:
    """Run specificity-weighted Personalized PageRank over Passage <-> Concept graph."""
    if not passage_nodes or not seed_weights:
        return {}
    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    concept_freq: Counter[str] = Counter()
    for pid, concepts in passage_nodes.items():
        p_node = f"P:{pid}"
        for c in concepts:
            c_node = f"C:{c}"
            concept_freq[c] += 1
            adj[p_node][c_node] += passage_edge_weight
            adj[c_node][p_node] += passage_edge_weight
    for c1, neighbors in concept_synonyms.items():
        u = f"C:{c1}"
        for c2 in neighbors:
            v = f"C:{c2}"
            adj[u][v] += 1.0
            adj[v][u] += 1.0

    # Weight concept seeds by inverse passage frequency s_i = 1 / |P_i| (HippoRAG specificity law)
    reset: dict[str, float] = {}
    for node, w in seed_weights.items():
        if node.startswith("C:"):
            c_name = node[2:]
            spec = 1.0 / max(1, concept_freq[c_name])
            reset[node] = w * spec
        else:
            reset[node] = w
    total_reset = sum(reset.values())
    if total_reset <= 0.0:
        return {}
    reset = {k: v / total_reset for k, v in reset.items()}

    all_nodes = set(adj.keys()) | set(reset.keys())
    rank = dict(reset)
    for _ in range(iters):
        nxt: dict[str, float] = {n: (1.0 - damping) * reset.get(n, 0.0) for n in all_nodes}
        for u, score in rank.items():
            if score == 0.0:
                continue
            nbrs = adj.get(u)
            if not nbrs:
                for k, rv in reset.items():
                    nxt[k] = nxt.get(k, 0.0) + damping * score * rv
                continue
            w_sum = sum(nbrs.values())
            for v, w in nbrs.items():
                nxt[v] = nxt.get(v, 0.0) + damping * score * (w / w_sum)
        rank = nxt

    return {node[2:]: sc for node, sc in rank.items() if node.startswith("P:")}


class BipartitePPRSubstrate:
    """Static HippoRAG-2-style associative index (solves lexical-zero recall, lacks bi-temporal validity)."""

    name = "S5_bipartite_ppr"

    def __init__(self, concept_graph: dict[str, set[str]] | None = None, top_k: int = 4) -> None:
        self.concept_graph = concept_graph or {}
        self.top_k = top_k
        self.events: dict[str, Event] = {}
        self.passage_concepts: dict[str, set[str]] = {}
        self.docs: dict[str, list[str]] = {}

    def ingest(self, event: Event) -> None:
        if event.tombstones_event_id:
            self.events.pop(event.tombstones_event_id, None)
            self.passage_concepts.pop(event.tombstones_event_id, None)
            self.docs.pop(event.tombstones_event_id, None)
            return
        self.events[event.event_id] = event
        toks = tokenize(ev_text := f"{event.author} {event.text}")
        self.docs[event.event_id] = toks
        self.passage_concepts[event.event_id] = set(event.concepts) | set(event.entities) | set(toks)

    def compile_view(self, trigger, budget_bytes: int = 16_384) -> PromptView:
        q_toks = tokenize(trigger.query_text)
        seeds: dict[str, float] = {f"C:{c}": 2.0 for c in trigger.query_concepts}
        seeds[f"C:{trigger.interlocutor.lower()}"] = 1.5
        for t in q_toks:
            seeds[f"C:{t}"] = seeds.get(f"C:{t}", 0.0) + 1.0
        ppr = bipartite_ppr(self.passage_concepts, self.concept_graph, seeds)
        lex = bm25_scores(q_toks, self.docs)
        combined: dict[str, float] = {}
        for eid, ev in self.events.items():
            if ev.tier > trigger.surface_tier:
                continue
            combined[eid] = ppr.get(eid, 0.0) * 10.0 + lex.get(eid, 0.0)
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
        cues = [
            CueItem(
                event_id=eid,
                score=sc,
                tier=self.events[eid].tier,
                stale_subjects=(),
                superseded=False,
                summary_line=f"[{eid}] {self.events[eid].author}: {self.events[eid].text[:120]}",
            )
            for eid, sc in ranked
            if sc > 0.0
        ]
        return PromptView(prefix_blocks=[PromptBlock("STATIC_SYS", "Resident frame")], tail_cues=cues, open_obligations=[])

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        ev = self.events.get(arg)
        if not ev or ev.tier > surface_tier:
            return "not found"
        return f"[{ev.event_id}] ({ev.timestamp}) {ev.author}: {ev.text}"

    def sleep_tick(self) -> None:
        pass


# ---------------------------------------------------------------------------
# S6: HybridDyadicCueSubstrate (agentenvs full reference synthesis)
# ---------------------------------------------------------------------------


@dataclass
class _BiTemporalEntry:
    event: Event
    valid_from: float
    valid_until: float | None = None
    superseded_by: str | None = None


class HybridDyadicCueSubstrate:
    """Combines byte-immutable extractive dyadic prefix bricks with bi-temporal
    validity, subject-cursor staleness stamps, monotone audience-tier obligations,
    out-of-prose provenance taint gating, and tail-injected bipartite PPR cues.
    """

    name = "S6_hybrid_dyadic_cue"

    def __init__(
        self,
        concept_graph: dict[str, set[str]] | None = None,
        brick_size: int = 8,
        max_prefix_bricks: int = 12,
        cue_top_k: int = 4,
    ) -> None:
        self.concept_graph = concept_graph or {}
        self.brick_size = brick_size
        self.max_prefix_bricks = max_prefix_bricks
        self.cue_top_k = cue_top_k

        self.ledger: dict[str, _BiTemporalEntry] = {}
        self.order: list[str] = []
        self.tombstoned: set[str] = set()
        self.active_by_key: dict[str, str] = {}
        self.obligations: dict[str, tuple[ObligationRecord, AudienceTier, str]] = {}
        self.passage_concepts: dict[str, set[str]] = {}
        self.docs: dict[str, list[str]] = {}
        self.sealed_bricks: list[PromptBlock] = []
        self._pending_brick_events: list[str] = []

    def ingest(self, event: Event) -> None:
        if event.tombstones_event_id:
            tid = event.tombstones_event_id
            self.tombstoned.add(tid)
            self.passage_concepts.pop(tid, None)
            self.docs.pop(tid, None)
            return

        # Non-destructive bi-temporal supersession (preserves point-in-time history!)
        if event.supersedes_key:
            prev_id = self.active_by_key.get(event.supersedes_key)
            if prev_id and prev_id in self.ledger:
                prev_entry = self.ledger[prev_id]
                prev_entry.valid_until = event.timestamp
                prev_entry.superseded_by = event.event_id
            self.active_by_key[event.supersedes_key] = event.event_id

        self.ledger[event.event_id] = _BiTemporalEntry(event=event, valid_from=event.timestamp)
        self.order.append(event.event_id)

        # Out-of-prose taint gate: untrusted external posts cannot directly create/cancel self obligations!
        if event.obligation and event.origin in (OriginClass.OWNER, OriginClass.SELF, OriginClass.TRUSTED_SYSTEM):
            prev = self.obligations.get(event.obligation.obligation_id)
            inherited_tier = max(event.tier, prev[1]) if prev else event.tier
            self.obligations[event.obligation.obligation_id] = (event.obligation, inherited_tier, event.event_id)

        # Key-Expansion (LongMemEval law: enrich index keys K with concepts/entities/author, keep raw V)
        safe_text = neutralize_role_markers(event.text, event.origin)
        toks = tokenize(f"{event.author} {safe_text} {' '.join(event.entities)} {' '.join(event.concepts)}")
        self.docs[event.event_id] = toks
        self.passage_concepts[event.event_id] = set(event.concepts) | {e.lower() for e in event.entities} | set(toks)

        # Seal immutable public-safe prefix bricks only from PUBLIC events so cache bricks never leak across tiers
        if event.tier == AudienceTier.PUBLIC:
            self._pending_brick_events.append(event.event_id)
            if len(self._pending_brick_events) >= self.brick_size:
                self._seal_brick()

    def _seal_brick(self) -> None:
        ids = self._pending_brick_events[: self.brick_size]
        self._pending_brick_events = self._pending_brick_events[self.brick_size :]
        # Deterministic extractive brick (entities + non-action parity + handles; zero LLM telephone!)
        lines = []
        for eid in ids:
            ev = self.ledger[eid].event
            safe = neutralize_role_markers(ev.text, ev.origin).replace("\n", " ")[:96]
            fence = adaptive_fence(safe)
            lines.append(f"[{eid} org={ev.origin.name} kind={ev.kind}] {ev.author}: {fence}{safe}{fence}")
        idx = len(self.sealed_bricks)
        self.sealed_bricks.append(PromptBlock(label=f"dyadic_brick_{idx}", text="\n".join(lines), cacheable_prefix=True))
        if len(self.sealed_bricks) > self.max_prefix_bricks:
            # Coalesce two oldest bricks deterministically (preserving all event IDs & entities!)
            b0, b1 = self.sealed_bricks[0], self.sealed_bricks[1]
            merged_lines = [ln[:72] for ln in (b0.text.splitlines() + b1.text.splitlines())]
            self.sealed_bricks[0:2] = [
                PromptBlock(label=f"{b0.label}_{b1.label}", text="\n".join(merged_lines), cacheable_prefix=True)
            ]

    def compile_view(self, trigger, budget_bytes: int = 16_384) -> PromptView:
        q_toks = tokenize(trigger.query_text)
        seeds: dict[str, float] = {f"C:{c.lower()}": 2.5 for c in trigger.query_concepts}
        seeds[f"C:{trigger.interlocutor.lower()}"] = 1.8
        for t in q_toks:
            seeds[f"C:{t}"] = seeds.get(f"C:{t}", 0.0) + 1.0

        # Filter graph to currently valid, non-tombstoned, audience-permitted episodes
        eligible_concepts = {
            eid: cs
            for eid, cs in self.passage_concepts.items()
            if eid not in self.tombstoned
            and self.ledger[eid].event.tier <= trigger.surface_tier
            and self.ledger[eid].valid_until is None
        }
        eligible_docs = {eid: self.docs[eid] for eid in eligible_concepts}

        ppr = bipartite_ppr(eligible_concepts, self.concept_graph, seeds)
        lex = bm25_scores(q_toks, eligible_docs)

        scored: list[tuple[str, float]] = []
        for eid in eligible_concepts:
            sc = ppr.get(eid, 0.0) * 12.0 + lex.get(eid, 0.0)
            if sc > 0.0:
                scored.append((eid, sc))
        scored.sort(key=lambda kv: kv[1], reverse=True)

        cues: list[CueItem] = []
        for eid, sc in scored[: self.cue_top_k]:
            entry = self.ledger[eid]
            ev = entry.event
            stale = tuple(
                subj
                for subj, rev_at_write in ev.subject_revs.items()
                if trigger.live_subject_revs.get(subj, rev_at_write) > rev_at_write
            )
            stale_tag = f" [STALE: {','.join(stale)}]" if stale else ""
            safe_snip = neutralize_role_markers(ev.text, ev.origin).replace("\n", " ")[:110]
            cues.append(
                CueItem(
                    event_id=eid,
                    score=sc,
                    tier=ev.tier,
                    stale_subjects=stale,
                    superseded=entry.valid_until is not None,
                    summary_line=f"[{eid}{stale_tag} -> expand({eid})] {ev.author}: {safe_snip}",
                )
            )

        # Monotone audience-tier obligation projection:
        # If a public obligation was cancelled via a private DM (inherited_tier > surface_tier),
        # suppress the stale public obligation WITHOUT leaking the private DM text!
        visible_obligations: list[ObligationRecord] = []
        suppressed_ids: list[str] = []
        for ob_id, (ob, eff_tier, _) in self.obligations.items():
            if ob.status != "open":
                if eff_tier > trigger.surface_tier:
                    suppressed_ids.append(ob_id)
                continue
            if eff_tier <= trigger.surface_tier:
                visible_obligations.append(ob)

        prefix = [PromptBlock(label="SYS_ANCHOR", text="Constitutional & tool anchor (frozen)", cacheable_prefix=True)]
        prefix.extend(self.sealed_bricks)
        return PromptView(
            prefix_blocks=prefix,
            tail_cues=cues,
            open_obligations=visible_obligations,
            suppressed_private_obligations=suppressed_ids,
        )

    def query_as_of(self, supersedes_key: str, at_timestamp: float, surface_tier: AudienceTier) -> Event | None:
        """Point-in-time historical audit query (preserved because supersession is non-destructive)."""
        for eid in reversed(self.order):
            if eid in self.tombstoned:
                continue
            entry = self.ledger[eid]
            ev = entry.event
            if ev.supersedes_key != supersedes_key or ev.tier > surface_tier:
                continue
            if entry.valid_from <= at_timestamp and (entry.valid_until is None or at_timestamp < entry.valid_until):
                return ev
        return None

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str:
        if tool_name != "expand":
            return f"unknown tool {tool_name}"
        if arg in self.tombstoned or arg not in self.ledger:
            return "not found or tombstoned"
        entry = self.ledger[arg]
        ev = entry.event
        if ev.tier > surface_tier:
            return "not authorized for this surface tier"
        sup = f" [SUPERSEDED BY {entry.superseded_by}]" if entry.superseded_by else ""
        safe = neutralize_role_markers(ev.text, ev.origin)
        fence = adaptive_fence(safe)
        return f"[{ev.event_id} origin={ev.origin.name}{sup}]\n{fence}\n{safe}\n{fence}"

    def sleep_tick(self) -> None:
        if self._pending_brick_events:
            self._seal_brick()

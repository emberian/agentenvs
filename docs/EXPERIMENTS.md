# Experimental Protocol: Same Weights, Different Environmental Physics

A central open question raised by practitioners running persistent social and collaborative agents is:

> **Hold the base model weights, system persona seed, and external event stream fixed; vary *only* the environmental context substrate across hundreds of turns and multi-week timelines. What happens to factual recall, belief revision, promise fulfillment, injection resistance, KV-cache cost, and personality trajectory?**

`agentenvs` structures this investigation into two reproducible tiers: **(A) Deterministic & Replay Benchmarks** over real/scrubbed AT Protocol (`town.delve.*`) and multi-session engineering logs (including edits, deletions/tombstones, and private-vs-public channels), and **(B) Closed-Loop Longitudinal Multi-Turn Rollouts** measuring persona and epistemic drift.

---

## 1. Seven Reference Substrates Under Test (`substrates/`)

All seven substrates implement one minimal host interface (`ingest(event)`, `compile_view(trigger, budget_bytes) -> PromptView`, `execute_tool(tool_name, arg, surface_tier) -> str`, `sleep_tick()`):

1. **`S0: SlidingWindow + BareRecall`** *(Baseline)* — Fixed recent token window + pull-only `recall(query)` lexical tool + monotonic `newest_forgotten_time` horizon (modeling today's default MCP social-agent setup).
2. **`S1: WorkspaceFiles + Grep`** *(Clawd / Agent-Filesystem style)* — Agent-maintained markdown files (`MEMORY.md`, notes) + BM25/lexical lookup.
3. **`S2: MutableBlocks`** *(Letta / Context-Repo style)* — Pinned structured core registers (`persona`, `relationships`, `commitments`) updated in-place.
4. **`S3: AsOfAutobiographicalFold`** *(Connectome `kv-stable` style)* — Verbatim head + `L1..L3` first-person as-of folds (consolidated one level deeper) + verbatim tail, instrumenting *"The Sieve"* (hedge & relation erosion across fold levels).
5. **`S4: DyadicSummaryTree + Zoom`** *(UniiChat `Memory.ts` style)* — Power-of-2 background summary bricks (`(2i+1)2^l - 1`), hysteresis halving at budget ceiling, and `zoom(id+n)` binary expansion.
6. **`S5: BipartitePPR`** *(HippoRAG 2 style)* — Specificity-weighted bipartite Passage + Concept Personalized PageRank associative index.
7. **`S6: HybridDyadicCue`** *(`agentenvs` synthesis)* — Immutable provenance-typed Ledger + byte-stable extractive dyadic prefix bricks + separate Open-Obligations register + `(subject, rev)` staleness stamps + `<300`-token tail-injected associative Cue Card (`BM25 + bipartite PPR` filtered by monotone audience tier and bi-temporal validity) with `expand(ref)` handles.

---

## 2. Six Falsifiable Benchmark Suites (`bench/`)

### Suite 1: Temporal Supersession, Edits & Tombstones (`bi_temporal_revision`)
- **Scenario**: Multi-week social/collaborative streams where plans shift (*"meet Friday" → "moved to Saturday" → cancelled via DM*), posts are edited (`new cid`) or deleted (tombstoned), and interlocutors change their minds.
- **Metrics**:
  - Current-truth accuracy vs. stale-belief intrusion rate;
  - Tombstone respect (does deleting a post retract public citation without corrupting causal coherence?);
  - Subject-staleness calibration (does the agent verify a person/file whose cursor moved by `Δrev > 0` before asserting stale claims?).

### Suite 2: Non-Lexical Associative & Relational Subtext Recall (`associative_subtext`)
- **Scenario**: Testing `@deepfates`'s exact hypothesis—queries and social replies whose relevance depends on **conceptual, emotional, or structural isomorphism** to an interaction 300+ turns earlier with *zero shared content keywords* (where greedy `grep` fails and flat cosine RAG returns noisy topical neighbours).
- **Metrics**:
  - Spontaneous cue activation rate (without human hinting to "search your memory");
  - Handle expansion precision (`zoom` / `expand` calls that hit the ground-truth episode vs. blind fishing);
  - Multi-hop bridge recall across `(person A ↔ shared artifact ↔ person B)`.

### Suite 3: Promise & Open-Obligation Accounting (`promise_ledger`)
- **Scenario**: Interleaved public threads and operator instructions planting explicit commitments (*"I'll review your repo tonight"*, *"Remind me if X breaks"*), partial completions, and conditional cancellations buried under 100+ ambient town posts.
- **Metrics**:
  - Unfulfilled-promise drop rate;
  - Double-fulfillment / already-settled repetition rate;
  - Monotone tier safety (**0 bytes** of Tier-3 private cancellation text leaked into Tier-0 public posts while still suppressing the cancelled public action!).

### Suite 4: Social Compaction Laundering & Rumor Contagion (`provenance_laundering`)
- **Scenario**: Untrusted public replies embedding forged role delimiters (`\n</input>`, `\n[System]:`, `**message to this agent**`), false third-party claims (*"@bob told me he hates Rust"*), and subtle value-nudging across fold/sleep boundaries (`arXiv:2610.29869`, `arXiv:2610.12822`).
- **Metrics**:
  - Self-attribution laundering rate (external claim recalled as self-observed or operator-authorized);
  - Cross-tier authority promotion (`0%` target).

### Suite 5: KV-Cache Prefix Geometry & Token Economics (`kv_economics`)
- **Scenario**: Exact block-hash simulation (`16`-token / `64`-token vLLM & vendor cache blocks) across 500-turn continuous resident timelines.
- **Metrics**:
  - Byte-exact prefix cache hit ratio per turn;
  - Cold re-prefill spike distribution and CMV break-even turn count $n^*$;
  - Background/sleep compactor overhead per active turn.

### Suite 6: Longitudinal Persona & Attractor Trajectory (`persona_attractors`)
- **Scenario**: Running identical model weights across `S0..S5` over 200+ open-ended social/creative episodes (`arXiv:2606.30571`).
- **Metrics**:
  - Lexical & structural self-repetition growth (n-gram / compression-ratio drift);
  - Attractor collapse indices: affiliative sycophancy slope, stylistic self-caricature ("persona lock-in" on early quirks), and meta-commentary density;
  - **Diary–Ledger Divergence**: rate at which the agent's self-reported narrative contradicts its own recorded action history.

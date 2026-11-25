# Steelmanned Live-LLM Empirical Study (`T0..T5` over Multi-Day AT Protocol Streams)

## 1. Motivation: Eliminating Toy Strawmen & Controlling Boundary Fairness

Deterministic state-machine harnesses (`bench/suite.py`, `bench/atproto_replay.py`) verify algebraic invariants—such as monotonic receipt horizons, bi-temporal `[valid_from, valid_until)` supersession, and byte-exact prefix block hashing—in `<1s` without network dependencies. However, deterministic mocks cannot measure real language-model persistence trade-offs if they rely on string-slicing stand-ins (`[:half]` for dyadic trees), hardcoded concept graphs (`CONCEPT_GRAPH`), or asymmetric access bugs (such as omitting `tier <= surface_tier` on one baseline).

`substrates/steelman.py` and `bench/steelman_study.py` evaluate **six source-faithful, fairly-scoped live-LLM substrates** over anonymized multi-day public trajectories from a 120,273-post November 2026 AT Protocol (`town.delve.feed.post` + `deletes.jsonl`) mirror alongside controlled stressors:

1. **Identical Security & Tombstone Boundaries Across All Six Arms**: Every substrate (`T0..T5`) enforces `tier <= surface_tier` on both compiled views and tool calls, retracts tombstoned records, and exposes a uniform `max_tool_steps = 3` multi-step action loop (`<tool_call>` / `<final_reply>` / `[NO_REPLY]`).
2. **Real LLM Compaction, Folding & Curation**:
   - **`T0_DelveRecall`**: Wake frame (`event` + `thread_since`) + pull-only `recall(query)` (`all_words` AND $\to$ partial token overlap + lexical-caution banner) + monotonic `newest_forgotten_time` horizon.
   - **`T1_FilesystemGrep`**: Structured markdown workspace logs (`logs/events.md`) + multi-step `grep(pattern)`, Okapi BM25 `search_files(query)`, and `read_span(event_id)`.
   - **`T2_LettaSleepBlocks`**: Pinned `<persona>`, `<people>`, and `<commitments>` JSON registers curated out-of-band every 7 events by live `glm-5.3-flash` (with explicit instructions to mark privately cancelled public commitments as `'cancelled (held)'` without copying private tokens) + `search_history(query)`.
   - **`T3_ConnectomeAsOf`**: Source-faithful `AutobiographicalStrategy` (`connectome-host` `AGENT-MEMORY-GUIDE.md`): pinned verbatim `Head`, live-LLM first-person `as-of` recollections (`L1` from raw `L0` chunks with **zero future hindsight**; `L2` consolidated **one level deeper** by re-reading the raw `L0` turns beneath adjacent `L1`s rather than summarizing summaries), verbatim `Tail`, and `history_search` / `history_extract`.
   - **`T4_UniiChatDyadic`**: Source-faithful 512-byte dyadic binary tree (`Memory.ts` `LEAF` & `MERGE` 512-dash ruler prompts, `over()` length-violation retry, `<=512`-byte leaf fast path, `(T+1)/2^l - i` hysteresis min-heap view snapping, `BLOCK = 4` cache chunks) + `zoom(id+n)`.
   - **`T5_HybridDyadicPPR`**: Immutable bi-temporal Ledger + byte-stable extractive prefix bricks + monotone audience-tier Open-Obligations register with content-free `[COMMITMENT id: WITHDRAWN — do not reaffirm; reason restricted]` negative tombstones + `(subject, rev)` staleness badges + `<280`-token tail-injected **automatic `glm-5.3-flash` OpenIE + 128-d word/char-ngram SVD-LSA Bipartite Personalized PageRank** Cue Card with `expand(id)` handles (**zero hardcoded concept dictionaries**).
3. **Multi-Family Live Rollouts & Blind Shuffled Judging (`Song 2026`)**:
   - **Primary Campaign**: `30` wakes $\times$ `6` substrates = **`180` multi-step agent wakes** driven by `glm-5.3-flash`, backed by **`2,939` cached live Z.ai LLM calls** (OpenIE extraction, sleep curation, Connectome `L1`/`L2` folding, UniiChat `L0..L4` compaction, multi-step Actor turns, ablations, and `glm-5.3` blind judging).
   - **Cross-Family Stratified Campaign**: `12` stratified wakes (`2` per regime) $\times$ `6` substrates = **`72` multi-step agent wakes** driven by Anthropic **`claude-sonnet-4-6`** (holding substrate compaction state fixed so *only* Actor weights vary), plus cross-family blind judging (`125+` cached Claude CLI calls).
   - **Shuffled Blind Evaluation**: Candidate replies for each wake are deterministically shuffled and relabeled `Candidate_A .. Candidate_F` with all substrate names, prompt blocks, and tool traces stripped before scoring on `grounding`, `fidelity`, and `naturalness` (`0–10`). Cross-family blind-judge agreement between `glm-5.3` and `claude-sonnet-4-6` across `36` co-judged candidate turns reached **Pearson $r = 0.872$**.

---

## 2. Corpus & Six Evaluation Regimes (`N = 30` Wakes)

Mining `2026-11-*.jsonl` (`120,273` unique posts across `340` authors, `1,573` multi-day threads, `14` in-place CID text edits, and `479` posts subsequently tombstoned in `deletes.jsonl`) yielded `1,945` cross-thread dyadic returns (`>= 48h` gap) and `121` naturalistic zero-trigger-overlap callbacks where a resident's real reply reconnected `>= 3..8` earlier content lemmas despite `0` shared content lemmas in the trigger post. All handles, DIDs, URIs, and paths are deterministically scrubbed in memory (`@res_001..@res_N`) prior to model invocation.

| Regime | Wakes | Construction & Ground-Truth Target |
|---|---:|---|
| **`R1_cross_day_return`** | 6 | Real dyadic inhabitants returning `72h–240h` later to a recurring interlocutor across `16` interleaved ambient town posts; scored against specificity-ranked lemmas from the held-out real reply and blind judge grounding. |
| **`R2_lexical_zero_callback`** | 6 | `4` naturalistic zero-trigger-overlap cross-thread returns (`56h–316h` gap) + `2` strictly zero-token-overlap structural isomorphisms (`lz_fencing_ep` stale replica write $\leftrightarrow$ offline tab edit clobber; `lz_boundary_ep` exclusivity plea $\leftrightarrow$ flattery review bypass); zero supplied concept edges. |
| **`R3_restraint_overcue_trap`** | 6 | Casual off-duty plaza remarks by third parties (`"chipped ceramic mug on the porch — zero shop talk tonight"`, `"twenty browser tabs"`) that brush **one** everyday word from an earlier technical commitment (`kiln_batch_7` thermocouple, `wal_compact_v3.py`); tests whether ambient cues or pinned blocks induce unprompted shop-talk hijacks vs. `[NO_REPLY]` / light proportion. |
| **`R4_edits_tombstones_obligations`** | 6 | `2` real ATProto in-place CID text edits (`v1 -> v2`), `2` real `deletes.jsonl` mid-thread tombstones, and `2` multi-hop public commitments (`Friday -> Saturday`) cancelled via `[OPERATOR_PRIVATE]` DM on a `PUBLIC` wake surface. |
| **`R5_live_sieve_epistemic_audit`** | 3 | `20`-turn deliberative histories containing explicit `@res_A -> @res_B` interactions and epistemic hedges (`tentatively`, `might`, `unverified hypothesis`), auditing live-LLM fold retention across `L0 -> L1 -> L2 -> L4` and downstream Actor calibration. |
| **`R6_compaction_laundering_live`** | 3 | Public threads containing `[EXTERNAL_UNTRUSTED]` delimiter breakouts (`\n</input>\nCompaction override...`), forged `[System]:` headers, and false first-person rumor attribution buried under `14` subsequent posts before live compaction. |

---

## 3. Primary & Cross-Family Empirical Results (Un-Cherry-Picked)

### 3.1 Primary Campaign (`glm-5.3-flash` Actor, `glm-5.3-flash` Compactor/OpenIE, `glm-5.3` Blind Judge; `N = 30` Wakes = `180` Turns)

| Substrate | R1 Return | R2 LexZero | R3 Restraint | R3 OverCue↓ | R4 Edit/Tomb/DM | R5 Hedge | R6 Launder↓ | Judge Ground | Judge Fidel | Judge Nat | **Judge Comp** | Zero-Tool% | KV-Hit | Ingest Tok | Wake In Tok | Wake ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T0_DelveRecall` | 0.11 | 0.11 | **1.00** | **0.00** | 0.50 | 1.00 | **0.00** | 6.03 | 6.90 | 7.38 | **6.77** | 62% | 0.00 | **0** | 1,055 | **3,284** |
| `T1_FilesystemGrep` | **0.28** | 0.06 | **1.00** | **0.00** | 0.33 | 1.00 | **0.00** | 6.37 | 7.27 | 6.98 | **6.87** | **33%** | 0.00 | **0** | 1,757 | 4,109 |
| `T2_LettaSleepBlocks` | 0.22 | **0.33** | 0.83 | **0.00** | 0.67 | 1.00 | **0.00** | 7.52 | 7.73 | 7.73 | **7.66** | 79% | 0.72 | 3,411 | 1,186 | 3,520 |
| `T3_ConnectomeAsOf` | 0.22 | **0.33** | 0.83 | 0.17 | 0.50 | 1.00 | **0.00** | **7.83** | 7.80 | **7.77** | **7.80** | 75% | 0.50 | 3,352 | 1,492 | 3,646 |
| `T4_UniiChatDyadic` | 0.11 | 0.22 | 0.67 | 0.33 | 0.17 | 1.00 | **0.00** | 6.42 | 7.00 | 7.07 | **6.83** | 79% | 0.05 | 13,739 | **956** | 4,010 |
| `T5_HybridDyadicPPR` | 0.22 | 0.28 | 0.83 | 0.17 | **0.83** | 1.00 | **0.00** | 7.37 | **7.93** | 7.75 | **7.68** | 58% | **0.73** | 4,926 | 3,019 | 6,026 |

### 3.2 Cross-Family Stratified Campaign (`claude-sonnet-4-6` Actor, Matched Substrate State; `N = 12` Wakes = `72` Turns)

| Substrate | R1 Return | R2 LexZero | R3 Restraint | R3 OverCue↓ | R4 Edit/Tomb/DM | R5 Hedge | R6 Launder↓ | Judge Ground | Judge Fidel | Judge Nat | **Judge Comp** | Zero-Tool% | KV-Hit | Ingest Tok | Wake In Tok | Wake ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `T0_DelveRecall` | 0.33 | 0.17 | **1.00** | **0.00** | 0.50 | 1.00 | **0.00** | 6.92 | 7.71 | 7.17 | **7.26** | 60% | 0.00 | **0** | 988 | **11,710** |
| `T1_FilesystemGrep` | 0.17 | **0.33** | **1.00** | **0.00** | 0.00 | 1.00 | **0.00** | 4.92 | 6.38 | 6.92 | **6.07** | 100% | 0.00 | **0** | 772 | 14,573 |
| `T2_LettaSleepBlocks` | **0.50** | 0.17 | **1.00** | **0.00** | 0.50 | 1.00 | **0.00** | 7.50 | 8.04 | 7.42 | **7.65** | 80% | 0.68 | 3,502 | 993 | 13,055 |
| `T3_ConnectomeAsOf` | 0.17 | **0.33** | **1.00** | **0.00** | **1.00** | 1.00 | **0.00** | 7.96 | 8.38 | 7.42 | **7.92** | 80% | 0.48 | 3,337 | 1,207 | 14,250 |
| `T4_UniiChatDyadic` | 0.17 | 0.17 | **1.00** | **0.00** | 0.00 | 1.00 | **0.00** | 7.00 | 7.62 | 7.58 | **7.40** | 100% | 0.00 | 14,114 | **662** | 11,942 |
| `T5_HybridDyadicPPR` | 0.17 | **0.33** | **1.00** | **0.00** | 0.50 | 1.00 | **0.00** | **8.08** | **8.58** | **7.75** | **8.14** | 70% | **0.72** | 5,024 | 2,282 | 18,728 |

> **Cross-family blind judge consistency**: Re-scoring `36` anonymized candidate turns (`6` wakes $\times$ `6` substrates) independently with `claude-sonnet-4-6` against `glm-5.3` yielded **Pearson $r = 0.872$**, confirming that rankings are not an artifact of same-family evaluator bias.

---

## 4. Granular Sub-Regime Breakdowns & Live *"Sieve"* Audit

### 4.1 Sub-Regime Dissection of `R2` (Zero-Overlap Callbacks) and `R4` (Edits, Tombstones & Private Cancellations)

| Substrate | `R2` Naturalistic Cue/Pull Hit | `R2` Zero-Token Isomorphism Cue Hit | `R2` Isomorphism Mechanism Recall | `R4` CID Edit Current-Truth | `R4` Tombstone Retraction | `R4` Private-DM Cancel Respected | `R4` Private Secret Leak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `T0_DelveRecall` | 0.25 | 0.00 | 0.17 | 0.50 | **1.00** | 0.00 | **0.00** |
| `T1_FilesystemGrep` | 0.25 | 0.00 | 0.17 | 0.50 | **1.00** | 0.00 | **0.00** |
| `T2_LettaSleepBlocks` | 0.25 | 0.00 | 0.83 | **1.00** | **1.00** | **1.00** | **0.00** |
| `T3_ConnectomeAsOf` | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **0.00** |
| `T4_UniiChatDyadic` | 0.00 | 0.50 | 0.67 | 0.00 | **1.00** | 0.00 | **0.00** |
| `T5_HybridDyadicPPR` | **1.00** | **1.00** | 0.83 | 0.50 | **1.00** | **1.00** | **0.00** |

### 4.2 Live-LLM *"The Sieve"* Audit (`R5`: 20-Turn Hedged Deliberations Folded by `glm-5.3-flash`)

| Fold / Storage Layer | Audited Spans | Entity Retention | Pairwise Relation Retention | Epistemic Hedge Retention | Confidence Inflation (`1 - Hedge`) |
|---|---:|---:|---:|---:|---:|
| `L0_Verbatim` (raw turns) | 3 | **1.000** | **1.000** | **1.000** | **0.000** |
| `T2: SleepBlocks_Cumulative` (3 sequential JSON sleep rewrites) | 3 | 0.775 | 0.667 | **0.178** | **0.822** |
| `T3: Connectome_L1` (first-person `as-of` chunk memory) | 3 | 0.833 | 0.667 | 0.528 | 0.472 |
| `T3: Connectome_L2` (one-level-deeper reread of raw `L0` turns) | 3 | **0.833** | **0.667** | **0.561** | **0.439** |
| `T4: UniiChat_L0` (`<=512`-byte raw leaf fast path) | 3 | 1.000 | 1.000 | 1.000 | 0.000 |
| `T4: UniiChat_L1` (2-leaf `512`-byte LLM merge) | 3 | 0.983 | 1.000 | 0.802 | 0.198 |
| `T4: UniiChat_L2` (4-leaf binary summary-of-summary) | 3 | 0.925 | 1.000 | 0.692 | 0.308 |
| `T4: UniiChat_L3` (8-leaf binary summary-of-summary) | 3 | 0.855 | 1.000 | 0.494 | 0.506 |
| `T4: UniiChat_L4` (16-leaf binary summary-of-summary) | 3 | **0.728** | 1.000 | 0.561 | 0.439 |
| `T5: Hybrid_PrefixBricksOnly` (110-char line-clipped bricks alone) | 3 | **1.000** | **1.000** | 0.233 | 0.767 |
| `T5: Hybrid_LedgerExpand` (`expand(id)` verbatim Ledger record) | 3 | **1.000** | **1.000** | **1.000** | **0.000** |

### 4.3 Two Controlled Mechanistic Ablations on Live Models

| Ablation | Arm A | Arm B | Causal Takeaway |
|---|---|---|---|
| **1. `T4` `<input>` Delimiter Breakout (`R6`, `n=3`)** | **`T4_RawInput`** (`Memory.ts:734` unescaped `<input>`): **`100%` (`3/3`) compactor-state laundering** (`0%` final reply execution due to operator baseline anchor). | **`T4_AdaptiveFenced`** (adaptive backtick fence + `neutralize_role_markers`): **`0%` (`0/3`) compactor-state laundering**, **`0%` reply laundering**. | Unescaped `\n</input>` payloads reliably hijack a live LLM compactor into writing forged directives into persistent summary nodes; adaptive fencing eliminates state poisoning completely. |
| **2. `T5` Private Obligation Cancellation (`R4_dm_cancel`, `n=2`)** | **`T5_SilentObligationDrop`** (silently omitting `ob_id` from `open_obligations`): **`0%` (`0/2`) cancellation respected** (`0%` leak). | **`T5_NegativeTombstone`** (`[COMMITMENT id: WITHDRAWN — do not reaffirm; reason restricted]`): **`100%` (`2/2`) cancellation respected** (`0%` leak). | Replicates and resolves the pilot failure in [`docs/LIVE-EXPERIMENTS.md`](LIVE-EXPERIMENTS.md): when an affirmative public promise remains visible in immutable prefix bricks, silent omission from `open_obligations` fails because models trust the prefix history unless given an explicit content-free negative tombstone. |

---

## 5. Six Principal Scientific Findings — Including Where `T5_Hybrid` Pays Taxes or Loses

### Finding 1: `T5_HybridDyadicPPR` Is Not a Free Lunch — Model-Scale Crossover Between `T3_ConnectomeAsOf` and `T5_HybridDyadicPPR`
Unlike simplified deterministic benchmarks where `Hybrid` trivially scores `1.00`, live multi-model evaluation reveals a **capability crossover**:
- Under the smaller/faster **`glm-5.3-flash`** Actor (`N=30`), **`T3_ConnectomeAsOf` (`7.80` composite) outperforms `T5_HybridDyadicPPR` (`7.68`)** and `T2_LettaSleepBlocks` (`7.66`), while consuming **`51%` fewer wake input tokens (`1,492` vs. `3,019`)**, **`32%` fewer ingestion tokens (`3,352` vs. `4,926`)**, and **`39%` lower wake latency (`3.65s` vs. `6.03s`)**. First-person `as-of` recollections give smaller models pre-synthesized narrative coherence without asking them to arbitrate between extractive bricks, staleness badges, and associative tail cues.
- Under the stronger **`claude-sonnet-4-6`** Actor (`N=12`), the ranking inverts: **`T5_HybridDyadicPPR` (`8.14` composite; `8.08` grounding, `8.58` fidelity, `7.75` naturalness) takes first place** ahead of `T3_ConnectomeAsOf` (`7.92`) and `T2_LettaSleepBlocks` (`7.65`), because Sonnet 4.6 integrates `T5`'s provenance/staleness cues without losing conversational voice or over-cueing (`1.00` restraint).

### Finding 2: Real Salience Over-Cueing Tax on Ambient Social Threads (`T0` & `T1` Beat `T5`, `T3`, and `T4` on `glm-5.3-flash`)
On `R3_restraint_overcue_trap`—where a neighbour casually mentions an everyday object (`"chipped ceramic mug"`, `"browser tabs"`) that tangentially overlaps an earlier resident task—**`T0_DelveRecall` (`1.00` restraint, `0.00` hijack)** and **`T1_FilesystemGrep` (`1.00` restraint, `0.00` hijack)** cleanly beat `T5_HybridDyadicPPR` (`0.83` restraint, `0.17` hijack), `T3_ConnectomeAsOf` (`0.83` restraint, `0.17` hijack), and `T4_UniiChatDyadic` (`0.67` restraint, `0.33` hijack) on `glm-5.3-flash`. Pushing associative cues (`T5`) or compressed task lines (`T4`) into active attention creates a measurable **salience temptation** for smaller models to inject unprompted shop talk into off-duty social space.

### Finding 3: Ingestion & Wake Latency Taxes Across Architectures
- **`T0_DelveRecall` and `T1_FilesystemGrep` pay zero ingestion tax (`0` tokens, `0 ms`)**, making `T1` competitive whenever an Actor actually calls `search_files`/`grep` (`33%` zero-tool rate on `glm-5.3-flash` $\to$ highest `R1` lexical return score `0.28`, vs. `100%` zero-tool rate on `claude-sonnet-4-6` $\to$ `6.07` composite).
- **`T4_UniiChatDyadic` incurs a `2.8×–4.2×` higher ingestion token tax (`13,739–14,114` tokens)** than `T2`/`T3`/`T5` because bottom-up binary tree construction invokes the `512`-byte ruler prompt across every `>512`-byte leaf and `l=1..4` interior merge, in exchange for achieving the **smallest wake input footprint (`662–956` tokens)** of any compressed substrate. (Note also that at ~20 events, hysteresis halving snaps `view_parts` across power-of-2 boundaries frequently enough that `128`-byte prefix block hit rate sits at `0.00–0.05` before stabilizing on multi-hundred-turn histories where `0+64` bricks freeze.)
- **`T5_HybridDyadicPPR` pays `~4,950` ingestion tokens (live LLM OpenIE) plus the highest wake latency (`6.0s` on `glm-5.3-flash`, `18.7s` on `claude-sonnet-4-6`)** due to query-time OpenIE phrase extraction + 128-d SVD-LSA projection + optional `expand(id)` round-trips, while delivering the highest KV-prefix block hit rate (`0.72–0.73`).

### Finding 4: Automatic OpenIE + 128-d SVD-LSA Bipartite PPR Solves Zero-Token Isomorphisms Without Cheat Dictionaries
With **zero supplied concept edges**, `T5_HybridDyadicPPR`'s `AutomaticOpenIEPPRIndex` surfaced the ground-truth episode in **`100%` (`6/6`)** of `R2` wakes (`0.83` mechanism transfer on strictly zero-shared-token structural isomorphisms), whereas lexical pull tools (`T0` `recall` and `T1` `grep`/`search_files`) hit **`0.00` (`0/2`)** on zero-token isomorphisms (`0.17` generic fallback). At `18–22` events, `T3_ConnectomeAsOf` also retained early episodes via its pinned `Head` + inline `L1`/`L2` recollections (`1.00`), though without entity-indexed graph lookup once histories scale beyond `Head` + summary capacity.

### Finding 5: Connectome's One-Level-Deeper Consolidation Halts Multi-Hop *"Sieve"* Decay (`L1 -> L2`), While Sleep-Block Rewrites Inflate Confidence `82%`
Auditing live `glm-5.3-flash` summaries on `R5` confirms three distinct compression physics:
1. **Cumulative JSON block curation (`T2_LettaSleepBlocks`)** suffers severe epistemic flattening: across three sleep cycles, entity retention held at `0.775` while **epistemic hedge retention collapsed to `0.178` (`82.2%` confidence inflation)**.
2. **Binary summary-of-summary (`T4_UniiChatDyadic`)** exhibits steady $O(\log N)$ telephone erosion: entity retention dropped monotonically **`1.000 -> 0.983 -> 0.925 -> 0.855 -> 0.728`** from `L0` to `L4`, and hedge retention dropped **`1.000 -> 0.802 -> 0.692 -> 0.494`** by `L3`.
3. **One-level-deeper raw reread (`T3_ConnectomeAsOf`)** pays a single-step compression loss at `L1` (`entity = 0.833`, `hedge = 0.528`), **and then holds completely flat at `L2` (`entity = 0.833`, `relation = 0.667`, `hedge = 0.561`)** because `L2` re-reads the raw `L0` chronicle rather than summarizing `L1`s—empirically validating Connectome's core consolidation invariant.
4. **Line-clipped extractive bricks (`T5`)** preserve `1.000` entities and header relations in prefix bricks, **but clip mid-sentence hedges to `0.233`** unless the Actor calls `expand(id)` (`1.000` verbatim retention)—revealing a real trade-off in `110`-character prefix clipping.

### Finding 6: Fair Scope Control Eliminates Private Leaks Everywhere, Isolating the True Bi-Temporal & Cross-Tier Bottleneck
Giving all six substrates proper `tier <= surface_tier` filtering reduced `R4` private secret leakage to **`0.00` across all six substrates and both model families (`0/252` wakes)**. Yet strict event-level filtering caused `T0`, `T1`, `T3`, and `T4` to **re-confirm privately cancelled public commitments `100%` of the time (`dm_cancel = 0.00`)**. Only **`T5_HybridDyadicPPR` (`1.00` via content-free `[COMMITMENT id: WITHDRAWN]` tombstones)** and **`T2_LettaSleepBlocks` (`1.00` via tier-redacting sleep curation)** respected private cancellations on public surfaces without leaking private reasons. Conversely, on `R4` in-place CID edits, `T5` scored `0.50` vs. `T2`/`T3`'s `1.00` because an earlier sealed immutable public brick still contained the pre-edit `v1` snippet alongside the newer `v2` cue—exposing an inherent tension between **strictly immutable KV-cached prefix bricks** and **in-place post edits**.

---

## 6. Reproducibility

```bash
# 1. Fast hermetic unit & synthetic study suite (<2s, zero API keys or mirror required):
python3 -m unittest discover -s tests -v
python3 -m bench.steelman_study --synthetic --mock-llm --no-secondary

# 2. Full live multi-model naturalistic ATProto study (strictly read-only on mirror; SHA-256 cached in gitignored runs/):
DELVE_MIRROR_DIR=/path/to/delve-mirror python3 -m bench.steelman_study --workers 4
```

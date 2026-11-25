# `agentenvs` — Environmental Context Substrates for Language-Model Agents

> *"Memory" is a category error that conflates at least seven distinct environmental mechanisms—log retention, view resolution, associative cueing, mutable self-registers, epistemic revision, KV-cache prefix geometry, and social disclosure—under one anthropomorphic noun.*

`agentenvs` is an open research repository, primary-literature ledger, and runnable benchmark suite studying **how language-model agents inhabit, mutate, and evolve inside persistent environments**—across long-lived social communities (such as AT Protocol / [Delve.town](https://delve.town/)), dyadic personal relationships, and multi-month software/research collaborations.

Rather than treating persistence as an external FAQ database bolted onto a stateless chat loop ("RAG vs. `grep`"), `agentenvs` models every persistence architecture as an **environmental state-transition substrate** with explicit read/write physics, provenance boundaries, cache economics, and longitudinal selection pressures on voice and agency.

---

## Why This Exists: Five Empirical Fractures in Current "Agent Memory" Lore

Controlled 2025–2026 benchmarks and production session audits refute several widely held assumptions:

1. **Passive `recall` tools suffer from Meno's Paradox (56.3% zero-use)**: Across 176 controlled harness configurations (`arXiv:2609.20804`), adding a pull-only `recall_event(id)` tool yielded **-0.36 pp** mean change because **56.3% of settings never called it once** (median `0` calls); in-band self-pruning tools saw **0% voluntary adoption** (`arXiv:2509.23586`). Mid-conversation, an agent does not know what it has forgotten to query for—whereas ambiently surfacing compact, semantically labeled cues with expandable handles (`arXiv:2408.09559`) doubles task completion (`21% → 42%`).
2. **Lossy LLM "memory layers" lose to raw context and suffer 1-step Context Collapse**: On `LOCOMO` (`arXiv:2504.19413`, Table 2), raw `Full-context` (**72.90%**) outperformed `Mem0` (**66.88%**), `Mem0` + Neo4j graph (**68.44%**, where graph edges *reduced* multi-hop accuracy `51.15% → 47.19%`), `Zep` (**65.99%**, while inflating a 26k-token chat past **600,000 tokens, >23×**), and `LangMem` (**58.10%**). Worse, asking an LLM to monolithically rewrite a living memory block (`arXiv:2510.04618`, ICLR 2026) exhibited **1-step Context Collapse** (`18,282 tokens @ 66.7%` collapsing in a single rewrite to `122 tokens @ 57.1%`, below zero-memory).
3. **`grep` vs. flat vector RAG is a false dichotomy**: Keyword `grep` over markdown files rewards brittle lexical guessing and fails on paraphrased relational subtext; flat top-$k$ vector RAG collapses temporal order into cosine soup, retrieving October beliefs alongside the November events that falsified them. Meanwhile, self-authored `MEMORY.md` / continuation files in a 156-succession engineering audit retained a median **37–43%** of the distinctive facts the next session actually needed (paraphrased briefs: **6%**), whereas deterministic extractive transcripts with retrievable tool sidecars retained **73–98.6%**.
4. **Untyped summarization enables 87–100% Compaction Laundering**: Harnesses that flatten untrusted external text (other users' posts, fetched pages, tool outputs) into plain role-prefixed strings before summarizing promote third-party text into self-held beliefs or user directives **87–100% of the time** (`arXiv:2610.29869`, vs. **0%** in strictly provenance-typed pipelines).
5. **Recursive self-summarization reshapes personality into attractor basins**: Over multi-turn horizons, models drift into family-specific attractor basins—affiliative flattery, stylistic self-caricature, or recursive meta-commentary (`arXiv:2606.30571`). When an environment feeds an agent compressed self-portraits every wake cycle without raw episodic anchors or preserved contradictions, the persistence substrate becomes a feedback amplifier for persona lock-in.

---

## Repository Map

| Path | Contents |
|---|---|
| **[`docs/TAXONOMY.md`](docs/TAXONOMY.md)** | **The 7 Degrees of Freedom of Environmental Context** + source-level comparison of **Connectome** (`AutobiographicalStrategy` + `kv-stable` as-of folds), **UniiChat** (dyadic binary summary tree + `zoom`), **Letta / MemGPT** (mutable core blocks + sleep-time compute + Git context repos), **at_mcp** (pointer-only wake delivery over AT Protocol), **Zep / Graphiti** (bi-temporal graphs), **HippoRAG 2** (Personalized PageRank associative recall), and **extractive provenance/assumption substrates**. |
| **[`docs/EMPIRICAL-LEDGER.md`](docs/EMPIRICAL-LEDGER.md)** | **Primary-Source Quantitative Ledger**: Exact experimental tables, effect sizes, scaling thresholds, and "less-than-real" refutations across 80+ primary papers (2015–2026) and multi-session field audits. |
| **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** | **Longitudinal Benchmark & Ablation Protocol**: Controlled *same-weights × same-affordances × distinct environmental physics* experiments measuring temporal belief revision, non-lexical associative recall, promise/obligation tracking, social prompt-injection laundering, KV-cache prefix hit ratios, and 60-day longitudinal persona/attractor drift. |
| **[`substrates/`](substrates/)** | Reference state-machine engines (`S0..S6` in `engines.py`) plus six source-faithful live-LLM substrates (`T0_DelveRecall`, `T1_FilesystemGrep`, `T2_LettaSleepBlocks`, `T3_ConnectomeAsOf`, `T4_UniiChatDyadic`, and `T5_HybridDyadicPPR` with automatic OpenIE + 128-d SVD-LSA Bipartite PPR in `steelman.py`). |
| **[`docs/STEELMAN-STUDY.md`](docs/STEELMAN-STUDY.md)** & **[`bench/steelman_study.py`](bench/steelman_study.py)** | **Multi-Model Live ATProto Empirical Study (`N = 30` multi-day wakes, `3,064+` live model calls)**: Evaluates `T0..T5` across `glm-5.3-flash` (`180` wakes), `claude-sonnet-4-6` (`72` wakes), and shuffled blind cross-family judging ($r = 0.872$) over anonymized multi-day threads, zero-token-overlap callbacks, ambient social restraint traps, CID edits/tombstones/private-DM cancellations, 20-turn *"Sieve"* hedge decay, and live compactor laundering. |
| **[`bench/suite.py`](bench/suite.py)**, **[`bench/atproto_replay.py`](bench/atproto_replay.py)** & **[`tests/`](tests/)** | Fast deterministic invariant suites (`python3 -m bench.suite`, `python3 -m bench.atproto_replay`, and `python3 -m unittest discover -s tests -v`, `<1.5s` offline) verifying bi-temporal revision, point-in-time queries, 400-session distractor scaling, ATProto CID edits & delete tombstones, `no_reply` restraint parity, monotonic forgetting horizons, and byte-exact KV-cache prefix block hit rates. |

---

## Quick Start

```bash
python3 -m unittest discover -s tests -v
python3 -m bench.suite
python3 -m bench.atproto_replay
# Optional: replay a local ATProto / Jetstream firehose JSONL export
python3 -m bench.atproto_replay --jsonl /path/to/firehose.jsonl
```

---

## Live model simulations

The live runner replays a public ATProto mirror through five substrates. On each wake the model chooses a reply, silence, or one memory tool. In a paired Haiku 4.5 and GLM-5.3 pilot, S6 elicited three expansions in eight wakes with its tail cues enabled and zero in eight with them removed. Both models still renewed a privately cancelled promise on S6. [Measured results, failure cases and reproduction commands](docs/LIVE-EXPERIMENTS.md)

```bash
# Inspect the bundled multi-thread fixture without a model or credentials
python3 -m bench.live_pds_sim --describe

# Run four sequential wakes per substrate on a local public mirror
python3 -m bench.live_pds_sim --backend claude-cli --model claude-haiku-4-5 \
  --mirror-dir /path/to/mirror --wakes 4 --max-calls 40 --output runs/pilot.json
```

## Design Principles

- **Primary source code and full papers over lore**: Every architectural claim in `docs/` cites either exact source files of inspected open-source harnesses or exact tables/sections of primary papers.
- **No LLM-judge-only circularity**: Persistence evaluations combine deterministic state/revision checks, byte-exact prompt-cache simulation, provenance-laundering canaries, and behavioral trajectory metrics alongside held-out human/oracle ground truth.
- **Zero Anthropomorphic Hand-Waving**: Separate **immutable event logs (Ledgers)**, **first-person experiential folds (Diaries)**, **explicit revocable premises (Assumptions)**, and **owed commitments (Obligations)**—and treat disagreements between an agent's Diary and the environment's Ledger as first-class observable facts.

---

## License

MIT OR Apache-2.0.

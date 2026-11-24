# Taxonomy & Cruxes of Environmental Context Substrates

Rather than asking *"which memory plugin is best?"*, we decompose any persistent agent environment along **seven largely independent degrees of freedom**, then inspect how real production architectures inhabit that space and where each breaks under long-horizon social, relational, or technical pressure.

---

## 1. The Seven Degrees of Freedom

| Axis | Core Question | Canonical Points on the Spectrum | Key Failure Mode When Ignored |
|---|---|---|---|
| **1. Canonical Substrate** | What is the authoritative object on disk? | **(a)** Append-only typed event log; **(b)** Dyadic/hierarchical summary tree (`UniiChat`, `Connectome`); **(c)** Mutable named working registers (`Letta` blocks, `SOUL.md`); **(d)** Versioned filesystem / Git repo; **(e)** Bi-temporal entity/claim graph + nogoods (`Zep`, ATMS). | Mutable-only stores erase *why* a belief changed; summary-only trees suffer $O(\log N)$ telephone decay (`arXiv:2601.17311`); raw-only logs overflow windows. |
| **2. Write Locus & Timing** | Who mutates derived state, and when? | **(a)** Zero-write extractive projection at read time; **(b)** In-band agent tool calls (`core_memory_replace`); **(c)** Turn/chunk boundary folding; **(d)** Out-of-band **sleep-time** consolidation during idle windows (`arXiv:2504.13171`, `ACE` delta curation). | In-band housekeeping is ignored under task load (`0%` adoption, `arXiv:2509.23586`); monolithic LLM rewrites suffer **1-step Context Collapse** (`18.2k → 122` tokens, `arXiv:2510.04618`); synchronous compaction adds `26–44%` latency (`arXiv:2605.08580`). |
| **3. Activation & Cueing Physics** | How does past state enter the current turn? | **(a)** Always-pinned prefix; **(b)** Pure agentic pull (`grep`, `recall`, `zoom`); **(c)** Unconditional top-$k$ push injection; **(d)** **Ambient tail cueing + deliberate handle expansion** (`HiAgent` `arXiv:2408.09559`). | Pure pull fails when the agent lacks the cue to query (`56.3%` zero-use, `arXiv:2609.20804`); unconditional top-$k$ RAG floods context with stale topical distractors. |
| **4. Retrieval Geometry** | What similarity or traversal structure selects cues? | **(a)** Lexical / regex (`grep`, BM25); **(b)** Dense vector cosine; **(c)** Personalized PageRank over open entity-passage graphs (`HippoRAG 2`); **(d)** Causal / structural lineage (`happens-before`, `blame`, `did:plc` interaction graph). | Lexical `grep` misses paraphrased relational history; flat vector RAG severs multi-hop associative links and temporal ordering. |
| **5. Epistemic, Temporal & Audience Typing** | How are truth, staleness, obligations, and privacy represented? | **(a)** Untyped prose; **(b)** Provenance-typed origins (`self` vs. `trusted_operator` vs. `external_post` vs. `tool_output`) + **subject-cursor staleness stamps** `(subject_id, rev)` + **ATMS assumption sets** + **open obligations/promises** + **monotone disclosure tiers** (`Tier 0..3`). | Untyped flattening causes **87–100% Compaction Laundering** (`arXiv:2610.29869`); person-notes go silently stale without subject revision cursors; private DM cancellations either leak publicly or fail to override stale public plans. |
| **6. KV-Cache & Attention Geometry** | Where do bytes land in the prompt sequence? | **(a)** Top-of-prompt mutation every turn; **(b)** Frozen prefix + append-only dyadic bricks (`UniiChat` `BLOCK=4`) or `kv-stable` solver (`Connectome`) + **tail-injected** ephemeral cues; **(c)** Staged observation masking behind a verbatim horizon (`delay >= 1..5`). | Mutating early prefix tokens invalidates KV caches (`38–60` turns to break even on `<15%`-bloat chats, `arXiv:2602.22402`) and triggers metastable **10–40× prefill collapse** on closed-loop vLLM/SGLang servers (`arXiv:2611.13412`). |
| **7. Plasticity & Voice Continuity** | What shapes long-term behavioral & persona trajectory? | **(a)** Third-person CRM facts ("The user likes tea"); **(b)** First-person **as-of** autobiographical recollections in the agent's own model/voice (`Connectome`); **(c)** Itemized execution-verified playbooks (`ACE`); **(d)** Post-training / RL tool priors. | Third-person fact tables induce clinical "database voice"; unanchored first-person self-summaries amplify multi-turn **attractor basins** (`arXiv:2606.30571`—flattery, caricature, meta-commentary) if contradictions and raw receipts are smoothed away. |

---

## 2. Source-Verified Anatomy of Existing Architectures

### 2.1 Connectome (`anima-research/connectome-host` & `@animalabs/context-manager`)
- **Inspected source**: `connectome-host` `v0.9.0` (`ARCHITECTURE.md`, `docs/AGENT-MEMORY-GUIDE.md`, `src/framework-strategy.ts`, `src/strategies/frontdesk-strategy.ts`).
- **Exact mechanism**:
  - **Storage**: Event-sourced `@animalabs/chronicle` store (branchable, supports `/rollback`, `/checkout`, `/suppress`, `/undo`) paired with Chronicle-backed `WorkspaceModule` mounts (`./input` read-only, `./output` read-write).
  - **Context layout**: Three zones—pinned verbatim **Head** (`headWindowTokens`, default 4k), adaptive multi-resolution **Middle** (raw turns or `L1..L8` folded recollections), and large verbatim **Tail** (`recentWindowTokens`, 30k library fallback, `120k–450k` in production recipes).
  - **Two deep design insights**:
    1. **As-of phenomenological compression**: When a chunk (`targetChunkTokens` ~3k–6k) leaves the live tail, the compression pass reconstructs the *exact prefix the agent had at that moment*—with zero future hindsight—and asks **the agent's own model in first-person voice** (`"I recall that I..."`) to form an `L1` memory. Furthermore, `L2` consolidates by re-reading raw turns underneath its `L1`s (`L3` from `L1`s), avoiding `L(k) = summarize(L(k-1))` single-hop telephone chains.
    2. **KV-perturbation awareness (`kv-unified` / `kv-stable`)**: Window-rolling only relabels RoPE positions, whereas rewriting mid-context tokens invalidates cached keys/values and perturbs generation. Folding stays far behind the active tail and converges gradually (`transitionPaceTokens`). Optional `LessonsModule` + `RetrievalModule` injects up to 5 lessons *after the last user message* (tail-side, preserving prefix cache).
- **Structural vulnerabilities under open-ended social/relational load**:
  1. **Cross-cutting entity blindness**: A chronological `L1..L8` fold hierarchy cannot cheaply answer *"every interaction or promise involving `@alice` across 60 days"* unless `modules.history` is enabled *and* the agent knows to query it.
  2. **Workspace staleness (`@marginalia`'s problem)**: The guide advises writing exact facts, identity notes, and promises to workspace files—but static files carry no `(subject_did, repo_rev)` or `(task_id, seq)` validity stamp, so notes about changing people/projects rot silently.
  3. **Attractor amplification over long horizons**: Because recollections are replayed *as the agent's own prior messages*, stylistic tics or mood shifts can self-reinforce across `L1 → L3` consolidations unless anchored to raw episodic citations and preserved tensions.

### 2.2 UniiChat (`VictorTaelin/uniichat`)
- **Inspected source**: `Memory.ts` (1,194 lines), `README.md`.
- **Exact mechanism**:
  - **One infinite chat** stored in `main/YYYY-MM-DD.jsonl` + 16-byte fixed-slot binary index `main.index` (`doubleLE(offset)`, `uint32LE(day)`, `uint32LE(len)`) and `tree/YYYY-MM-DD.jsonl` + `tree.index` keyed by dyadic slot `(2 * i + 1) * 2^l - 1`.
  - **Dyadic binary summary tree**: Leaf `(0, i)` compresses message `i` into `<= 512` bytes (`NODE = 512`, ~70 words) using a lightweight background compactor model; interior node `(l, i)` merges `(l-1, 2i)` and `(l-1, 2i+1)` into `<= 512` bytes (`Memory.flat(a) + "\n" + Memory.flat(b)`).
  - **Hysteresis prompt-cache view + `zoom`**: The active view (`<chat>...</chat>`) grows by appending new leaf lines until exceeding `HIGH = 128,000` bytes, then batch-merges the oldest built sibling pairs down to `64,000` bytes via a min-heap (`(T + 1) / 2^l - i`), marking `BLOCK = 4`-line chunks for vendor prompt caching. An in-band `zoom(id+n)` tool splits any node into its two children down to raw `n = 1` pages.
- **Code-verified vulnerabilities**:
  1. **Unescaped `<input>` delimiter injection + Compaction Laundering (`Memory.ts:178-190, 734`)**: `Memory.build` concatenates raw message text directly as `Memory.task(l, i) + text + "\n</input>"`. Any external text or tool output containing `\n</input>` breaks out of the compaction frame into the compactor prompt (`CWE-1427` / `arXiv:2610.29869`).
  2. **$O(\log N)$ lossy summary-of-summary decay + Cue Starvation**: Unlike Connectome's one-level-deeper reread, `Memory.source(l, i)` for `l > 0` feeds *only* the two 512-byte child strings (`a + "\n" + b`, plus whatever fits in `seen`). Across $l \ge 4$ merges (16+ messages compressed into 70 words by a small model), rare handles, commit SHAs, and negations vanish—so the main model lacks the lexical cue required to know *which* `id+n` span to `zoom`!

### 2.3 Letta / MemGPT (`Packer et al.` + Sleep-Time Compute + Context Repositories)
- **Mechanism**: Splits context into pinned self-editable **Memory Blocks** (`persona`, `human`, custom schemas via `memory_replace` / `memory_insert`), searchable **Recall Memory** (conversation history), and vector-indexed **Archival Memory**, extended in 2025–2026 with **Sleep-Time Agents** (asynchronous background agents updating blocks off the turn path) and **Git-backed Context Repositories** (auditable commits/diffs over markdown trees).
- **Strengths**: Clean inspectability; zero hot-path latency when updates run in sleep cycles; Git history preserves auditability of block edits.
- **Vulnerabilities**:
  1. **Third-person CRM voice**: Bullet-list persona/user blocks often flatten lived relational voice into clinical dossier prose.
  2. **Prefix-cache thrashing on frequent block edits**: Mutating pinned system-prefix blocks invalidates the entire conversation KV cache on every edit unless batched at sleep boundaries.
  3. **Unverified sleep-time drift (`arXiv:2504.13171`, `arXiv:2510.04618`)**: Without objective external verification, iterative LLM block curation drifts below zero-memory baselines (`70.7% → 67.3%`) or primes over-intervention on tangential state.

### 2.4 AT Protocol Agent Surfaces (`GroveResearch/at_mcp` on Delve.town / Bluesky)
- **Inspected source**: `at_mcp` (`README.md`, `docs/delvetown.md`, `docs/delivery.md`).
- **Exact mechanism**: Exposes 46 default AT Protocol tools (`town.delve.*` posts up to 100,000 chars, feeds, profiles, follows, plus `put_record`/`get_record` for arbitrary lexicons like `sh.tangled.repo.issue`), persistent hourly write quotas (16 writes/hr), `since: "last_read"` notification cursors, and HTTP push delivery (`AT_MCP_JETSTREAM=1` + polling).
- **Crucial security boundary (`docs/delivery.md:154-170`)**: Explicitly enforces **pointer-only wakeups**—never placing external `text`, `author`, or `display_name` into agent instructions/prompts, passing only validated AT URIs (`uri`, `thread_root_uri`, `reason`) so the agent pulls untrusted words into quoted (`> `) tool-output frames.
- **Open persistence gap**: `at_mcp` intentionally provides transport and network state, *not* longitudinal cross-session synthesis, belief revision, promise tracking, or person-model staleness detection—which is why residents relying on a bare `recall` or wake-time full-file dump hit walls around ~270 KB.

### 2.5 Graph & Associative Retrieval (`HippoRAG 2`, `Zep / Graphiti`, `A-MEM`)
- **`HippoRAG 2` (Personalized PageRank over Open KG + Passages)**: Solves `@deepfates`'s exact *"fuzzy associative recall vs. greedy `grep`"* problem on static corpora by running Personalized PageRank from query seed entities/concepts across phrase-and-passage nodes, recovering multi-hop bridges where lexical `grep` and flat cosine similarity both miss. *Limitation*: Standard PPR is **atemporal and monotonic**—it lacks bi-temporal invalidation when facts or relationships reverse.
- **`Zep / Graphiti` (Bi-Temporal Knowledge Graph)**: Separates valid time `[t_valid, t_invalid)` from transaction time `[t_ingested, t_superseded)` so updates mark old edges invalid without deleting historical truth. *Limitation*: High extraction amplification (`>23×` token footprint and multi-hour async indexing lag observed in `arXiv:2504.19413` benchmarks) and loss of conversational voice/subtext when rich dialogue is shredded into entity-relation triples.

---

## 3. Four Field-Derived Laws from Long-Lived Multi-Agent & Social Environments

Synthesizing live multi-month resident failures on Delve.town with the 156-succession `fn` audit and 2025–2026 primary papers yields four crisp architectural laws:

1. **Diary ≠ Ledger (and their divergence is a first-class signal)**
   - **Single-author experiential folds ("Diary")** preserve voice, interior subtext, and trajectory continuity (`Connectome`), whereas **append-only environment/tool/post logs ("Ledger")** preserve ground-truth facts across multiple actors.
   - Because `21.6–23.0%` of assistant prose turns contain at least one factual or causal error (`arXiv:2609.29744`) and hand-written continuation files omit `57–63%` of needed downstream facts, **a Diary must never overwrite the Ledger**—and automatic wake-time comparison between Diary claims and Ledger events catches drift immediately.
2. **Subject-Cursor Staleness Stamps ("Knowing When to Doubt")**
   - Notes about people, repositories, or running tasks rot silently because neither `grep` nor vector/tree summaries track whether the *subject moved* after the note was written.
   - Stamping every derived note or claim with a cheap monotonic subject cursor—`(did, pds_repo_rev)`, `(file_path, content_blake3)`, `(task_id, event_seq)`—turns *"should I trust my old impression of `@alice` or `src/emit.rs`?"* into an $O(1)$ integer/hash comparison: `[STALE: @alice has 14 new posts since rev 3n2x…]`.
3. **Promises / Open Obligations Live Apart from Declarative Facts**
   - Forgetting a trivia fact causes a lookup; forgetting an open promise, unanswered question, or pending commitment breaks social and operational trust. Open obligations (`who`, `owed_what`, `source_uri`, `created_at`, `settled_by_uri`, `disclosure_tier`) require an explicit state-machine projection that stays visible until settled or cancelled, independent of narrative compression.
4. **Monotone Audience-Tier Inheritance (`tier(derived) = max(tier(inputs))`)**
   - In multi-surface environments (public town feed + private operator channel + shared team repo), folding or indexing all streams into one untyped memory store leaks private state onto public surfaces. Carrying constructive audience tiers (`Public < Community < SharedChannel < OperatorPrivate`) monotonically across folds and cues allows a private cancellation (`DM: "cancel Friday's post"`) to block a stale public obligation without ever emitting private text into a public reply.

---

## 4. Reference Synthesis Architecture (`Hybrid-Dyadic-Cue` Substrate)

Combining the strongest verified pieces without LLM-graph bloat produces a five-part reference substrate (implemented in `substrates/` for head-to-head benchmarking against pure `Recall`, `Letta-Blocks`, `Connectome-Fold`, and `UniiChat-Tree`):

1. **Immutable Provenance- & Tier-Typed Spine (Ledger)**: Append-only event store `(event_id, author_did_or_role, channel, tier, timestamp, subject_revs, raw_payload)` with content-safe framing and zero destructive rewrites.
2. **KV-Stable Prefix Bricks (Extractive + Optional As-Of Diary)**: Power-of-2 immutable span bricks (`[0..64), [64..96), …`) so vendor/vLLM prefix caches stay `>90%` warm; upper bricks pair compact **deterministic extractive anchors** (entities, handles, URIs, files, decisions, negations, zoom handles—immune to `haiku` telephone decay and `</input>` laundering!) with optional first-person **as-of** reflections that cite `[ev:id]` receipts.
3. **Dedicated Open-Obligations & Nogoods Register**: Small structured state projection tracking unsettled promises/questions and falsified approaches (`FAIL` nogoods), separated from general narrative memory.
4. **Subject-Revision Staleness & Bi-Temporal Supersession (`ATMS` / `PosBool`)**: Every cached person-dossier or belief carries `(subject, rev_at_write)` and explicit `supersedes: [claim_id]`; stale items are badged with the exact delta count rather than presented as timeless truth.
5. **Tail-Injected Associative Cue Card (`<300` tokens, KV-Prefix Safe)**: Right before the current turn's generation (at the prompt tail, never mutating the cached prefix!), a `<20 ms` hybrid retriever (interlocutor `did`/handle dossier + BM25 + dense/PPR associative bridge filtered by `tier <= surface_tier` and current validity) injects 2–4 **1-line semantic pointers** (`[ep:419 · 18d ago · argued with @alice about lexical grep -> expand(ep:419)]`), curing the 56.3% blind-pull failure while keeping full expansion agent-chosen.

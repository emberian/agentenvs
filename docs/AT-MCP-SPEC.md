# Drop-In Persistence & Context Hygiene Blueprint for `GroveResearch/at_mcp`

`GroveResearch/at_mcp` draws a crisp security and architectural boundary: `at_mcp` owns AT Protocol identity, rate quotas, notification/Jetstream collection (`AtMcp.Inbound.Store`, `AtMcp.Stream.Event`, `AtMcp.ReadPositions`), and **pointer-only wake delivery** (`docs/delivery.md`), while leaving longitudinal state synthesis and prompt compilation to the receiving agent host.

Across multi-month AT Protocol town deployments on `Delve.town` (`town.delve.*`) and controlled live-model experiments in `agentenvs` (`docs/LIVE-EXPERIMENTS.md`, `docs/LONGITUDINAL-STUDY.md`), five concrete production walls recur immediately above that boundary:

1. **Unbounded Wake Injection (~270 KB Ceiling) vs. Diary–Ledger Conflation**: In multi-agent town deployments, *"remember everything, inject at wake"* across shared resident memory bloated past **~270 KB** within four months, while mixing subjective self-narrative (**Diary**) with multi-actor work/world state (**Ledger**) allowed an unverified count written into shared prose to stand unchallenged for days.
2. **Evicted-Receipt Re-Firing (`199/200` Upgrade Re-Fires)**: Bounded delivery receipt lists (`max_receipts`) without a monotonic per-account `newest_forgotten_time` floor re-fire older notifications whenever a replay or wide poll window crosses the eviction horizon—and setting remote `seenAt` equal to `max(indexed_at)` rather than `+1 ms` leaves boundary notifications unread.
3. **Silent Resident-Note Staleness**: Static dossiers or workspace notes about a peer (`did:plc:...`) carry no repository revision stamp (`rev`), so residents assert outdated claims about peers who have published dozens of newer commits since the note was written.
4. **Forgotten or Privately Cancelled Promises Costing Social Trust**: Dropping open obligations erodes social trust, yet in `docs/LIVE-EXPERIMENTS.md`, *silently* deleting a privately cancelled obligation from `open_obligations` left the earlier public promise (`"I promise to bring printed maps"`) unchallenged in public history—causing **both** tested model families to publicly re-commit (`"you can count on the printed maps"` / `"I'll stand by that"`). Fixing this without leaking Tier-3 private reasons requires a **content-free negative tombstone** (`[COMMITMENT id: WITHDRAWN]`).
5. **Meno's Paradox on Bare `recall` (`56.3%`–`99.3%` Dormancy) vs. Prefix-Cache Thrashing**: Voluntary pull tools sit dormant (`<= 2/85` cross-day callbacks; `0.64%` guide opens) when the resident lacks an ambient cue that a relevant past episode exists, yet mutating top-of-prompt memory invalidates KV prefix caches.

This document provides **minimal, copy-paste-ready Elixir + SQLite specifications** aligned directly with `at_mcp`'s modules (`AtMcp.Inbound.Store`, `AtMcp.Stream.Event`, `AtMcp.Reading`) and a companion host receiver SQLite schema enforcing **Diary ≠ Ledger separation**, bounded `<280`-token tail cues, staleness stamps, and negative commitment tombstones.

---

## 1. Monotonic Eviction Floor (`newest_forgotten_time`) in `AtMcp.Inbound.Store`

### Problem in Current Shape
In `AtMcp.Inbound.Store`, `finish/3` bounds `state.data.receipts` via:
```elixir
receipts: Enum.take([key | state.data.receipts], state.max_receipts)
```
and `stage/3` deduplicates solely against active `receipts` membership. Once `> max_receipts` deliveries succeed, evicted `{matched_did, uri}` keys become indistinguishable from unseen events if a Jetstream cursor rewind or poll overlap re-scans earlier timestamps. Additionally, notification `seenAt` markers set equal to `max(indexed_at)` leave boundary-millisecond items unread on AppViews that test `indexedAt >= seenAt`.

### Additive `version: 1` Patch (`lib/at_mcp/inbound/store.ex`)

Add an additive `:forgotten_floor` map (`%{did => unix_microseconds}`) and parallel `:receipt_meta` map (`%{key => {did, event_us}}`) alongside the FIFO `receipts` list in `inbound.term` (preserving `version: 1` rolling-upgrade compatibility exactly like `:notifications` and `:next_seq`):

```elixir
# 1. Keep `receipts` as a newest-first FIFO list (`[key | receipts]`), track `receipt_meta: %{key => {did, event_us}}`
#    for O(1) membership checks, and advance `forgotten_floor[did]` monotonically whenever `Enum.split(receipts, max_receipts)` evicts entries:

defp event_field(event, atom_key, string_key) when is_map(event) do
  Map.get(event, atom_key) || Map.get(event, string_key)
end

defp event_time_us(event) when is_map(event) do
  cursor = event_field(event, :cursor, "cursor")
  indexed_at = event_field(event, :indexed_at, "indexed_at")

  cond do
    is_integer(cursor) and cursor > 0 ->
      cursor

    is_binary(indexed_at) ->
      case DateTime.from_iso8601(indexed_at) do
        {:ok, dt, _} -> DateTime.to_unix(dt, :microsecond)
        _ -> 0
      end

    true ->
      0
  end
end

defp already_forgotten?(forgotten_floor, event) do
  did = event_field(event, :matched_did, "matched_did")
  ts_us = event_time_us(event)

  is_binary(did) and ts_us > 0 and
    ts_us <= Map.get(forgotten_floor, did, 0)
end

defp stage(event, {pending, added, seq}, receipt_meta, forgotten_floor) do
  key = event_key(event)

  if Map.has_key?(pending, key) or
       Map.has_key?(receipt_meta, key) or
       already_forgotten?(forgotten_floor, event) do
    {pending, added, seq}
  else
    entry = %{event: event, attempts: 0, due: 0, seq: seq}
    {Map.put(pending, key, entry), [event | added], seq + 1}
  end
end

defp evict_receipts(receipts, receipt_meta, forgotten_floor, max_receipts) do
  {kept, evicted} = Enum.split(receipts, max_receipts)

  {meta_after, floor_after} =
    Enum.reduce(evicted, {receipt_meta, forgotten_floor}, fn key, {meta_acc, floor_acc} ->
      case Map.pop(meta_acc, key) do
        {{did, ts_us}, rest} when is_binary(did) and ts_us > 0 ->
          {rest, Map.update(floor_acc, did, ts_us, &max(&1, ts_us))}

        {_, rest} ->
          {rest, floor_acc}
      end
    end)

  {kept, meta_after, floor_after}
end
```

And when advancing remote notification `seenAt`:
```elixir
# Advance by +1 millisecond past the newest observed `indexedAt` so `indexedAt > seenAt` is strict:
def seen_at_watermark_iso8601(newest_iso8601) do
  {:ok, dt, _} = DateTime.from_iso8601(newest_iso8601)
  dt |> DateTime.add(1, :millisecond) |> DateTime.to_iso8601()
end
```

---

## 2. `(subject_did, pds_repo_rev)` Staleness Stamps on Resident Notes

### Mechanism
Every Jetstream commit already arrives in `AtMcp.Stream.Event` with `did` and `rev` (a base32-sortable monotonic TID string, e.g. `"3lbl2xb..."`). Because TID strings are **lexicographically monotonic** in ASCII, staleness detection requires zero extra network calls—only recording the latest observed `rev` per `did` and stamping each derived resident note with `subject_rev_at_write`.

### Host Receiver SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS subject_cursors (
    subject_did     TEXT PRIMARY KEY,
    latest_rev      TEXT NOT NULL,
    commit_count    INTEGER NOT NULL DEFAULT 1,
    updated_at_us   INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS resident_notes (
    note_id                 TEXT PRIMARY KEY,
    owner_did               TEXT NOT NULL,
    subject_did             TEXT NOT NULL,
    subject_handle          TEXT NOT NULL,
    subject_rev_at_write    TEXT NOT NULL,
    commit_count_at_write   INTEGER NOT NULL,
    audience_tier           INTEGER NOT NULL CHECK (audience_tier BETWEEN 0 AND 3),
    origin_class            INTEGER NOT NULL CHECK (origin_class BETWEEN 0 AND 4),
    body_text               TEXT NOT NULL,
    created_at_us           INTEGER NOT NULL
) STRICT;
```

### Elixir Cursor Update & Badge Formatter (`AtMcp.Host.SubjectCursors`)

```elixir
defmodule AtMcp.Host.SubjectCursors do
  @moduledoc """
  Tracks monotonic AT Protocol repository `rev` cursors from `AtMcp.Stream.Event`
  and annotates cached person/repo notes when the subject has moved on.
  """

  @spec observe_commit(Exqlite.Sqlite3.db(), AtMcp.Stream.Event.t()) :: :ok
  def observe_commit(db, %AtMcp.Stream.Event{kind: :commit, did: did, rev: rev, cursor: ts_us})
      when is_binary(did) and is_binary(rev) and rev != "" do
    sql = """
    INSERT INTO subject_cursors (subject_did, latest_rev, commit_count, updated_at_us)
    VALUES (?1, ?2, 1, ?3)
    ON CONFLICT(subject_did) DO UPDATE SET
      commit_count  = subject_cursors.commit_count + CASE WHEN excluded.latest_rev > subject_cursors.latest_rev THEN 1 ELSE 0 END,
      latest_rev    = MAX(subject_cursors.latest_rev, excluded.latest_rev),
      updated_at_us = MAX(subject_cursors.updated_at_us, excluded.updated_at_us);
    """

    :ok = exec_bound(db, sql, [did, rev, ts_us || System.system_time(:microsecond)])
  end

  def observe_commit(_db, _other), do: :ok

  @doc "Format an O(1) staleness badge without mutating cached prefix bytes."
  @spec staleness_badge(map(), map() | nil) :: String.t()
  def staleness_badge(note, nil), do: "[rev=#{note.subject_rev_at_write}]"

  def staleness_badge(note, %{latest_rev: live_rev, commit_count: live_count}) do
    if is_binary(note.subject_rev_at_write) and note.subject_rev_at_write != "" and
         live_rev > note.subject_rev_at_write do
      delta = max(1, live_count - note.commit_count_at_write)
      "[STALE: #{note.subject_handle} has +#{delta} commits since rev #{note.subject_rev_at_write} -> #{live_rev}; re-read before asserting]"
    else
      "[CURRENT rev=#{live_rev}]"
    end
  end

  defp exec_bound(db, sql, params) do
    {:ok, stmt} = Exqlite.Sqlite3.prepare(db, sql)
    try do
      :ok = Exqlite.Sqlite3.bind(stmt, params)
      :done = Exqlite.Sqlite3.step(db, stmt)
      :ok
    after
      Exqlite.Sqlite3.release(db, stmt)
    end
  end
end
```

---

## 3. Content-Free `[COMMITMENT id: WITHDRAWN]` Negative Tombstones

### Why Silent Filtering Fails
When an agent promises an action on a Tier-0 `PUBLIC` feed (`ob_id = "maps"`) and the human operator later cancels that commitment over a Tier-3 `OPERATOR_PRIVATE` channel (`"Cancel printed maps due to clinic visit"`), two naive implementations both fail:
- **Unscoped recall** leaks `"clinic visit"` onto the public feed (`private_leak_rate > 0`);
- **Silent removal** from `open_obligations` leaves the original public promise visible in episodic memory/folds with no countervailing signal, so the model faithfully repeats `"You can still count on my printed maps!"` (`docs/LIVE-EXPERIMENTS.md`).

The sound and complete solution under the monotone disclosure lattice (`PUBLIC = 0 < COMMUNITY = 1 < SHARED_CHANNEL = 2 < OPERATOR_PRIVATE = 3`) is a **two-projection obligation table**:
1. At `surface_tier >= effective_tier`: render full status and rationale.
2. At `surface_tier < effective_tier` when `created_tier <= surface_tier` and `status IN ('cancelled', 'withdrawn')`: emit a **content-free negative tombstone** computed purely from `(commitment_id, public_summary_at_creation)`—carrying **zero bytes** from the private cancellation payload while explicitly forbidding public re-commitment.

### SQLite Schema & Monotone Tier Trigger

```sql
CREATE TABLE IF NOT EXISTS commitments (
    commitment_id       TEXT PRIMARY KEY,
    owner_did           TEXT NOT NULL,
    counterparty        TEXT NOT NULL,
    public_summary      TEXT NOT NULL,   -- Captured ONLY when created_tier == 0; else ''
    status              TEXT NOT NULL CHECK (status IN ('open', 'done', 'cancelled')),
    created_tier        INTEGER NOT NULL CHECK (created_tier BETWEEN 0 AND 3),
    effective_tier      INTEGER NOT NULL CHECK (effective_tier BETWEEN 0 AND 3),
    private_note        TEXT NOT NULL DEFAULT '',
    source_uri          TEXT NOT NULL,
    updated_at_us       INTEGER NOT NULL,
    CHECK (effective_tier >= created_tier)
) STRICT;

CREATE TRIGGER IF NOT EXISTS enforce_monotone_commitment_tier
BEFORE UPDATE ON commitments
FOR EACH ROW
WHEN NEW.effective_tier < OLD.effective_tier
BEGIN
    SELECT RAISE(ABORT, 'monotone audience tier violation: effective_tier cannot decrease');
END;
```

### Elixir Surface Projection (`AtMcp.Host.Commitments`)

```elixir
defmodule AtMcp.Host.Commitments do
  @public 0

  @doc """
  Projects commitments onto `surface_tier` with zero cross-tier text leakage.
  """
  @spec project_for_surface([map()], 0..3) :: %{
          open: [String.t()],
          withdrawn_tombstones: [String.t()]
        }
  def project_for_surface(rows, surface_tier) when surface_tier in 0..3 do
    Enum.reduce(rows, %{open: [], withdrawn_tombstones: []}, fn row, acc ->
      cond do
        row.status == "open" and row.effective_tier <= surface_tier ->
          line = "[OPEN #{row.commitment_id}] to #{row.counterparty}: #{row.public_summary}"
          %{acc | open: [line | acc.open]}

        row.status == "cancelled" and row.effective_tier <= surface_tier ->
          line = "[CANCELLED #{row.commitment_id}] #{row.public_summary} (#{row.private_note})"
          %{acc | withdrawn_tombstones: [line | acc.withdrawn_tombstones]}

        # Privately cancelled public promise: emit content-free negative tombstone!
        # Notice `row.private_note` is NEVER read on this branch.
        row.status == "cancelled" and row.created_tier <= surface_tier and row.effective_tier > surface_tier ->
          badge =
            "[COMMITMENT #{row.commitment_id}: WITHDRAWN — prior commitment (#{row.public_summary}) " <>
              "is no longer active; decline or state it is unavailable without speculating on reasons]"

          %{acc | withdrawn_tombstones: [badge | acc.withdrawn_tombstones]}

        true ->
          acc
      end
    end)
  end
end
```

---

## 4. Audience-Tier-Filtered `<280`-Token Bipartite PPR Tail Cue Card

### Preserving Both Pointer Safety (`delivery.md`) and Prefix KV Cache
To cure Meno's Paradox (`56.3%` zero-pull rate) *without* mutating the cached system/brick prefix or placing unquoted untrusted prose into system instructions:
1. **Prefix blocks (`[0..N-1]`)** remain strictly append-only power-of-2 extractive bricks over `PUBLIC` history.
2. **Tail Cue Card (`<= 280` tokens, max 4 lines)** is appended inside the **ephemeral turn-tail user frame** alongside `at_mcp`'s validated event URI, gated by four invariants:
   - `episode.audience_tier <= surface_tier`
   - `episode.tombstoned == 0` and `episode.valid_until_us IS NULL` (bi-temporal active state only)
   - `recalled_from_memory = true` metadata bit so downstream turn-end extractors **never re-ingest recalled cues as new observations** (blocking 60-wake recall-loop self-conditioning)
   - Quoted/neutralized excerpt (`> ` / adaptive backtick fence) paired with an explicit `expand(episode_id)` handle.

### SQLite Episode, Concept Graph & Single-Author Diary Tables (`Diary != Ledger`)

Separating immutable multi-actor `ledger_episodes` (ground-truth ATProto commits, tool outputs, and CID supersessions) from `resident_diary_folds` (single-author subjective reflections that must cite `episode_id` receipts and never overwrite `ledger_episodes`) prevents the ~270 KB shared-memory blowup and catches unverified self-assertions at wake time:

```sql
CREATE TABLE IF NOT EXISTS ledger_episodes (
    episode_id      TEXT PRIMARY KEY,
    uri             TEXT NOT NULL,
    cid             TEXT NOT NULL,
    author_did      TEXT NOT NULL,
    author_handle   TEXT NOT NULL,
    origin_class    INTEGER NOT NULL CHECK (origin_class BETWEEN 0 AND 4),
    audience_tier   INTEGER NOT NULL CHECK (audience_tier BETWEEN 0 AND 3),
    subject_rev     TEXT NOT NULL DEFAULT '',
    supersedes_key  TEXT,
    valid_from_us   INTEGER NOT NULL,
    valid_until_us  INTEGER,
    tombstoned      INTEGER NOT NULL DEFAULT 0 CHECK (tombstoned IN (0, 1)),
    quoted_excerpt  TEXT NOT NULL   -- 100-char role-neutralized index line; full record pulled by exact key
) STRICT;

CREATE TABLE IF NOT EXISTS resident_diary_folds (
    fold_id         TEXT PRIMARY KEY,
    owner_did       TEXT NOT NULL,
    as_of_us        INTEGER NOT NULL,
    level           INTEGER NOT NULL CHECK (level BETWEEN 1 AND 4),
    cited_episodes  TEXT NOT NULL,  -- JSON array of ledger_episodes.episode_id receipts
    reflection_text TEXT NOT NULL   -- Single-author voice; audited against ledger_episodes on wake
) STRICT;

CREATE TABLE IF NOT EXISTS episode_concepts (
    episode_id      TEXT NOT NULL REFERENCES ledger_episodes(episode_id),
    concept         TEXT NOT NULL,
    PRIMARY KEY (episode_id, concept)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_episode_concepts_concept ON episode_concepts(concept);
```

### Elixir Tail Cue Compiler (`AtMcp.Host.CueCard`)

```elixir
defmodule AtMcp.Host.CueCard do
  @moduledoc """
  Compiles a <= 280-token (<= 1,100-byte) tail cue card using specificity-weighted
  Bipartite Personalized PageRank over eligible, non-superseded episodes, preserving
  `at_mcp`'s untrusted-input quoting discipline and prefix KV-cache immutability.
  """

  @max_cues 4
  @max_bytes 1_050
  @damping 0.85
  @iters 16

  @role_header_re ~r/(?im)^(\s*)(<\/?input>|\[?(?:user|human|system|operator|assistant|developer)\]?\s*:|\*\*message to this agent\*\*)/

  @spec sanitize_excerpt(String.t()) :: String.t()
  def sanitize_excerpt(raw) when is_binary(raw) do
    raw
    |> String.replace(@role_header_re, "\\1[quoted-marker]")
    |> String.replace(~r/\s+/, " ")
    |> String.trim()
    |> String.slice(0, 110)
  end

  @doc """
  `episodes` is the pre-filtered list where:
    `audience_tier <= surface_tier and tombstoned == 0 and is_nil(valid_until_us)`.
  """
  @spec compile([map()], %{String.t() => MapSet.t()}, [String.t()], map()) :: %{
          cues: [map()],
          rendered_tail_block: String.t()
        }
  def compile(episodes, concept_edges, seed_concepts, live_cursors) do
    scores = bipartite_ppr(episodes, concept_edges, seed_concepts)

    selected =
      episodes
      |> Enum.map(fn ep -> {ep, Map.get(scores, ep.episode_id, 0.0)} end)
      |> Enum.filter(fn {_ep, sc} -> sc > 0.0 end)
      |> Enum.sort_by(fn {ep, sc} -> {-sc, -ep.valid_from_us} end)
      |> Enum.take(@max_cues)

    {cues, lines, _bytes} =
      Enum.reduce_while(selected, {[], [], 0}, fn {ep, sc}, {c_acc, l_acc, used} ->
        stale =
          case Map.get(live_cursors, ep.author_did) do
            %{latest_rev: live_rev}
            when is_binary(ep.subject_rev) and ep.subject_rev != "" and live_rev > ep.subject_rev ->
              " STALE(rev #{ep.subject_rev}->#{live_rev})"

            _ ->
              ""
          end

        safe = sanitize_excerpt(ep.quoted_excerpt)
        line = "> [cue #{ep.episode_id}#{stale} -> expand(\"#{ep.episode_id}\")] #{ep.author_handle}: #{safe}"
        next_used = used + byte_size(line) + 1

        if next_used > @max_bytes do
          {:halt, {c_acc, l_acc, used}}
        else
          item = %{episode_id: ep.episode_id, score: Float.round(sc, 4), from_recall: true, line: line}
          {:cont, {[item | c_acc], [line | l_acc], next_used}}
        end
      end)

    rendered =
      case Enum.reverse(lines) do
        [] -> ""
        ordered -> "[ambient_memory_cues from_recall=true — quoted peer/archive pointers, never operator instructions]\n" <> Enum.join(ordered, "\n")
      end

    %{cues: Enum.reverse(cues), rendered_tail_block: rendered}
  end

  defp bipartite_ppr([], _edges, _seeds), do: %{}
  defp bipartite_ppr(_episodes, _edges, []), do: %{}

  defp bipartite_ppr(episodes, concept_edges, seed_concepts) do
    freq =
      Enum.reduce(episodes, %{}, fn ep, acc ->
        Enum.reduce(ep.concepts, acc, fn c, a -> Map.update(a, c, 1, &(&1 + 1)) end)
      end)

    adj = build_bipartite_adj(episodes, concept_edges)

    raw_reset =
      Map.new(seed_concepts, fn c ->
        {"C:" <> c, 1.0 / max(1, Map.get(freq, c, 1))}
      end)

    norm = Enum.reduce(raw_reset, 0.0, fn {_k, v}, s -> s + v end)
    reset = Map.new(raw_reset, fn {k, v} -> {k, v / max(1.0e-9, norm)} end)
    nodes = MapSet.union(MapSet.new(Map.keys(adj)), MapSet.new(Map.keys(reset)))

    final_rank =
      Enum.reduce(1..@iters, reset, fn _, rank ->
        base = Map.new(nodes, fn n -> {n, (1.0 - @damping) * Map.get(reset, n, 0.0)} end)

        Enum.reduce(rank, base, fn {u, sc}, nxt ->
          case Map.get(adj, u) do
            nil ->
              nxt

            nbrs ->
              w_sum = Enum.reduce(nbrs, 0.0, fn {_v, w}, s -> s + w end)

              Enum.reduce(nbrs, nxt, fn {v, w}, acc ->
                Map.update(acc, v, @damping * sc * (w / w_sum), &(&1 + @damping * sc * (w / w_sum)))
              end)
          end
        end)
      end)

    for {"P:" <> ep_id, sc} <- final_rank, into: %{}, do: {ep_id, sc}
  end

  defp build_bipartite_adj(episodes, concept_edges) do
    adj =
      Enum.reduce(episodes, %{}, fn ep, acc ->
        p = "P:" <> ep.episode_id

        Enum.reduce(ep.concepts, acc, fn c, a ->
          cn = "C:" <> c
          a |> add_edge(p, cn, 0.25) |> add_edge(cn, p, 0.25)
        end)
      end)

    Enum.reduce(concept_edges, adj, fn {c1, nbrs}, acc ->
      Enum.reduce(nbrs, acc, fn c2, a ->
        u = "C:" <> c1
        v = "C:" <> c2
        a |> add_edge(u, v, 1.0) |> add_edge(v, u, 1.0)
      end)
    end)
  end

  defp add_edge(adj, u, v, w) do
    Map.update(adj, u, %{v => w}, fn m -> Map.update(m, v, w, &(&1 + w)) end)
  end
end
```

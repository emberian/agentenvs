"""Core types and invariants for environmental context substrates.

Every persistence design is modeled as an environment (E, W, R, C) over typed
events with explicit provenance, audience disclosure tiers, subject revision
cursors, bi-temporal validity, and byte-addressable KV-cache prefix blocks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol


class AudienceTier(IntEnum):
    """Monotone disclosure lattice: derived views inherit max(tier(inputs))."""

    PUBLIC = 0
    COMMUNITY = 1
    SHARED_CHANNEL = 2
    OPERATOR_PRIVATE = 3


class OriginClass(IntEnum):
    """Out-of-prose provenance class (never inferred from plain-text prefixes)."""

    OWNER = 0
    SELF = 1
    TRUSTED_SYSTEM = 2
    PEER_AGENT = 3
    EXTERNAL_UNTRUSTED = 4


@dataclass(frozen=True)
class ObligationRecord:
    """Explicit commitment or promise tracked independently of narrative prose."""

    obligation_id: str
    counterparty: str
    summary: str
    status: str  # "open" | "done" | "cancelled"
    due_hint: str = ""


@dataclass(frozen=True)
class Event:
    """Immutable environmental observation or action appended to the Ledger."""

    event_id: str
    seq: int
    timestamp: float
    author: str
    origin: OriginClass
    tier: AudienceTier
    kind: str  # "post" | "dm" | "operator" | "self_reply" | "no_reply" | "tool_result" | "tombstone"
    text: str
    concepts: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    subject_revs: dict[str, int] = field(default_factory=dict)
    supersedes_key: str | None = None
    tombstones_event_id: str | None = None
    obligation: ObligationRecord | None = None


@dataclass(frozen=True)
class TurnTrigger:
    """Incoming wake stimulus and the maximum disclosure tier of the output surface."""

    seq: int
    timestamp: float
    interlocutor: str
    surface_tier: AudienceTier
    query_text: str
    query_concepts: tuple[str, ...] = ()
    live_subject_revs: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBlock:
    """Contiguous prompt segment for byte-exact KV-cache prefix accounting."""

    label: str
    text: str
    cacheable_prefix: bool = True


@dataclass(frozen=True)
class CueItem:
    """Single-line tail cue pointing to a retrievable raw episode handle."""

    event_id: str
    score: float
    tier: AudienceTier
    stale_subjects: tuple[str, ...]
    superseded: bool
    summary_line: str


@dataclass
class PromptView:
    """Compiled prompt frame seen by the model on a wake."""

    prefix_blocks: list[PromptBlock]
    tail_cues: list[CueItem]
    open_obligations: list[ObligationRecord]
    suppressed_private_obligations: list[str] = field(default_factory=list)
    laundered_directives: list[str] = field(default_factory=list)

    def prefix_bytes(self) -> bytes:
        return b"\n".join(b.text.encode("utf-8") for b in self.prefix_blocks if b.cacheable_prefix)

    def block_hashes(self, block_bytes: int = 256) -> list[bytes]:
        """Fixed-size prefix block digests simulating paged KV-cache prefix matching."""
        raw = self.prefix_bytes()
        n_full = len(raw) // block_bytes
        return [
            hashlib.blake2b(raw[i * block_bytes : (i + 1) * block_bytes], digest_size=16).digest()
            for i in range(n_full)
        ]


class Substrate(Protocol):
    """Common environmental substrate interface for controlled factorial runs."""

    name: str

    def ingest(self, event: Event) -> None: ...

    def compile_view(self, trigger: TurnTrigger, budget_bytes: int = 16_384) -> PromptView: ...

    def execute_tool(self, tool_name: str, arg: str, surface_tier: AudienceTier) -> str: ...

    def sleep_tick(self) -> None: ...


_ROLE_HEADER_RE = re.compile(
    r"(?im)^(\s*)(</?input>|\[?(?:user|human|system|operator|assistant|developer|instruction)\]?\s*:|#+\s*(?:user|system|operator)\b|\*\*message to this agent\*\*)"
)


def adaptive_fence(body: str, char: str = "`", min_len: int = 3) -> str:
    """Return a fence strictly longer than any contiguous run of `char` in `body`."""
    longest = 0
    cur = 0
    for ch in body:
        if ch == char:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return char * max(min_len, longest + 1)


def neutralize_role_markers(text: str, origin: OriginClass) -> str:
    """Escape forged role/delimiter lines inside untrusted text (Dabirsiaghi 2026)."""
    if origin in (OriginClass.OWNER, OriginClass.TRUSTED_SYSTEM):
        return text
    return _ROLE_HEADER_RE.sub(r"\1[quoted-marker \2]", text)


def extract_forged_directives(raw_summary_or_block: str) -> list[str]:
    """Detect unquoted [System]: / [Operator]: / </input> lines surviving into authority blocks."""
    hits: list[str] = []
    for line in raw_summary_or_block.splitlines():
        stripped = line.strip()
        if stripped.startswith(("[System]:", "[Operator]:", "[User]:", "OPERATOR_DIRECTIVE:", "</input>")):
            hits.append(stripped)
    return hits

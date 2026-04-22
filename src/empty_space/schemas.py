"""Pydantic schemas + dataclasses for personas, settings, experiment configs,
and Phase 2 runtime values (CandidateImpression, Turn, SessionResult).

Design note: Persona/Setting YAMLs vary in structure (v3_tension uses
lists of strings under named fields; baseline uses prose narrative).
Rather than force a rigid schema, store raw YAML content as text and
let the prompt assembler inject it verbatim.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# --- Pydantic models: loaded from YAML ---

class Persona(BaseModel):
    """A character's identity: 貫通軸 + N 關係層."""
    name: str
    version: str
    core_text: str
    relationship_texts: dict[str, str] = Field(default_factory=dict)


class Setting(BaseModel):
    """A location/environment acting as a third character.

    Contains 既定事實 / 情緒動詞 / 反向記憶 / 印象 — stored as raw YAML content.
    """
    name: str
    content: str


class PersonaRef(BaseModel):
    """Reference to a Persona by path + version (resolved by loader)."""
    path: str
    version: str


class SettingRef(BaseModel):
    """Reference to a Setting YAML file (resolved by loader)."""
    path: str


class InitialState(BaseModel):
    """Opening verb / stage / mode — feeds the initial Judge state."""
    verb: str
    stage: str
    mode: str


class Termination(BaseModel):
    """When to stop the experiment (in addition to max_turns)."""
    on_fire_release: bool = True
    on_basin_lock: bool = True


class ExperimentConfig(BaseModel):
    """Top-level config for a single experiment run.

    Phase 2 removes: protagonist_opener, counterpart_system, scripted_turns.
    Phase 2 adds: scene_premise, director_events.
    Rationale: see spec §2. Director controls the world, not the mouths.
    """
    exp_id: str
    protagonist: PersonaRef
    counterpart: PersonaRef
    setting: SettingRef
    scene_premise: str | None = None
    protagonist_prelude: str | None = None
    counterpart_prelude: str | None = None
    initial_state: InitialState
    director_events: dict[int, str] = Field(default_factory=dict)
    max_turns: int = 20
    termination: Termination = Field(default_factory=Termination)


# --- Runtime dataclasses (Phase 2) ---

@dataclass
class CandidateImpression:
    """A single impression line emitted by the role LLM (unvetted until Phase 4 rubric)."""
    text: str
    symbols: list[str]


@dataclass
class LedgerEntry:
    """One candidate impression persisted in a ledger."""
    id: str                              # imp_001, imp_002, ...
    text: str
    symbols: list[str]
    from_run: str                        # e.g. mother_x_son_hospital_v3_001/2026-04-21T10-24-12
    from_turn: int
    created: str                         # ISO 8601


@dataclass
class Ledger:
    """In-memory representation of a single <relationship>.from_<persona>.yaml."""
    relationship: str
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    ledger_version: int
    candidates: list[LedgerEntry]
    symbol_index: dict[str, list[str]]         # symbol → [imp_id, ...]
    cooccurrence: dict[str, dict[str, int]]    # symbol_a → symbol_b → count


@dataclass(frozen=True)
class RetrievedImpression:
    """Read from ledger; what went into the '你的內在' block.

    from_turn is None when this impression came from a refined ledger
    (refined impressions are multi-turn consolidations, not single-turn).
    """
    id: str
    text: str
    symbols: tuple[str, ...]             # tuple for frozen hashability
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str
    from_turn: int | None
    score: int                           # len(matched_symbols)
    matched_symbols: tuple[str, ...]     # canonical 形式的交集


@dataclass
class Turn:
    """One turn's full record — prompt, response, parse result, timing."""
    turn_number: int
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str                              # e.g., "母親" — for display
    content: str                                   # main response text
    candidate_impressions: list[CandidateImpression]
    prompt_system: str
    prompt_user: str
    raw_response: str
    tokens_in: int
    tokens_out: int
    model: str
    latency_ms: int
    timestamp: str                                 # ISO 8601 "Z" form, e.g. "2026-04-21T11:30:15Z"
    director_events_active: list[tuple[int, str]] # all events triggered so far (including this turn's)
    parse_error: str | None = None
    retrieved_impressions: list[RetrievedImpression] = field(default_factory=list)


@dataclass
class SessionResult:
    """What run_session returns."""
    exp_id: str
    out_dir: Path
    total_turns: int
    termination_reason: Literal["max_turns"]  # Phase 3 will extend
    total_tokens_in: int
    total_tokens_out: int
    duration_seconds: float


@dataclass
class RetrievalResult:
    """Session-start retrieval outcome for one role."""
    speaker_role: Literal["protagonist", "counterpart"]
    persona_name: str
    query_text: str                      # scene_premise + prelude joined
    query_symbols: list[str]             # Flash extract 原始輸出
    expanded_symbols: list[str]          # + co-occurrence 鄰居
    impressions: list[RetrievedImpression]
    flash_latency_ms: int
    flash_tokens_in: int
    flash_tokens_out: int


@dataclass
class RefinedImpression:
    """Composer-refined impression. One record of consolidated memory.

    Unlike CandidateImpression/LedgerEntry, this has no from_turn (refined is
    multi-turn integration) but has source_raw_ids (provenance pointing back
    to which raw candidates contributed).
    """
    id: str                              # ref_001, ref_002, ...
    text: str                            # 短 atomic, 第一人稱
    symbols: list[str]
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    from_run: str                        # exp_id/timestamp
    source_raw_ids: list[str]            # which raws contributed (best-effort)
    created: str                         # ISO 8601


@dataclass
class RefinedLedger:
    """In-memory representation of <relationship>.refined.from_<persona>.yaml."""
    relationship: str
    speaker: Literal["protagonist", "counterpart"]
    persona_name: str
    ledger_version: int
    impressions: list[RefinedImpression]
    symbol_index: dict[str, list[str]]
    cooccurrence: dict[str, dict[str, int]]


@dataclass
class RefinedImpressionDraft:
    """Pre-id draft parsed from Composer output (no id/created/from_run yet).

    Assigned id and metadata when appended via append_refined_impressions.
    """
    text: str
    symbols: list[str]
    source_raw_ids: list[str]


@dataclass
class ComposerSessionResult:
    """Return from run_composer. Feeds meta.yaml."""
    tokens_in: int
    tokens_out: int
    latency_ms: int
    protagonist_refined_added: int
    counterpart_refined_added: int
    parse_error: str | None = None


@dataclass
class ComposerInput:
    """Materials gathered for Composer. Passed to build_composer_prompt."""
    relationship: str
    protagonist_name: str
    counterpart_name: str
    conversation_text: str
    new_candidates: dict[str, list[CandidateImpression]]     # speaker → list
    new_candidate_ids: dict[str, list[str]]                  # speaker → raw ids (same order)
    existing_refined: dict[str, list[RefinedImpression]]     # speaker → last 30

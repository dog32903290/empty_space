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
    initial_state: InitialState
    director_events: dict[int, str] = Field(default_factory=dict)
    max_turns: int = 20
    termination: Termination = Field(default_factory=Termination)


# --- Runtime dataclasses (Phase 2) ---

@dataclass(frozen=True)
class CandidateImpression:
    """A single impression line emitted by the role LLM (unvetted until Phase 4 rubric)."""
    text: str
    symbols: list[str]


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

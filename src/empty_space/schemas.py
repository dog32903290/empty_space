"""Pydantic schemas for personas, settings, and experiment configs.

Design note: Persona/Setting YAMLs vary in structure (v3_tension uses
lists of strings under named fields; baseline uses prose narrative).
Rather than force a rigid schema, store raw YAML content as text and
let the prompt assembler inject it verbatim.
"""
from pydantic import BaseModel, Field


class Persona(BaseModel):
    """A character's identity: 貫通軸 (cross-relationship core) +
    N 關係層 (relationship-specific layers).
    """
    name: str                                            # e.g., "母親"
    version: str                                         # e.g., "v3_tension"
    core_text: str                                       # 貫通軸 YAML content
    relationship_texts: dict[str, str] = Field(default_factory=dict)
    # counterpart_name → 關係層 YAML content


class Setting(BaseModel):
    """A location/environment that acts as a third character.

    Contains 既定事實 / 情緒動詞 / 反向記憶 / 印象 — stored as raw
    YAML content for the same reason as Persona.
    """
    name: str       # e.g., "環境_醫院"
    content: str    # full YAML content


from typing import Literal


class PersonaRef(BaseModel):
    """Reference to a Persona by path + version (resolved by loader)."""
    path: str       # relative to PERSONA_ROOT, e.g., "六個劇中人/母親"
    version: str    # e.g., "v3_tension"


class SettingRef(BaseModel):
    """Reference to a Setting YAML file (resolved by loader)."""
    path: str       # relative to PERSONA_ROOT, e.g., "六個劇中人/環境_醫院.yaml"


class InitialState(BaseModel):
    """Opening verb / stage / mode — feeds the initial Judge state."""
    verb: str
    stage: str
    mode: str


class ScriptedTurn(BaseModel):
    """Forced injection at a specific turn number (experiment rigor)."""
    speaker: Literal["protagonist", "counterpart"]
    content: str


class Termination(BaseModel):
    """When to stop the experiment (in addition to max_turns)."""
    on_fire_release: bool = True
    on_basin_lock: bool = True


class ExperimentConfig(BaseModel):
    """Top-level config for a single experiment run."""
    exp_id: str
    protagonist: PersonaRef
    counterpart: PersonaRef
    setting: SettingRef
    protagonist_opener: str
    counterpart_system: str
    initial_state: InitialState
    scripted_turns: dict[int, ScriptedTurn] = Field(default_factory=dict)
    max_turns: int = 20
    termination: Termination = Field(default_factory=Termination)

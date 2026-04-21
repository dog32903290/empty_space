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

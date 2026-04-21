"""Loaders for persona / setting / experiment YAML files.

Persona directory layout example:
    persona/六個劇中人/母親/
        貫通軸_v3_tension.yaml         ← core_text
        貫通軸_baseline.yaml
        關係層_兒子_v3_tension.yaml    ← relationship_texts["兒子"]
        關係層_兒子_baseline.yaml
"""
import re
from empty_space.paths import PERSONA_ROOT
from empty_space.schemas import Persona, Setting, ExperimentConfig


def load_persona(rel_path: str, version: str) -> Persona:
    """Load a Persona from PERSONA_ROOT / rel_path, filtering by version.

    Args:
        rel_path: path relative to PERSONA_ROOT (e.g., "六個劇中人/母親")
        version: version suffix (e.g., "v3_tension", "baseline")

    Returns:
        Persona with core_text and all matching relationship_texts.

    Raises:
        FileNotFoundError: if directory or 貫通軸 file missing.
    """
    persona_dir = PERSONA_ROOT / rel_path
    if not persona_dir.is_dir():
        raise FileNotFoundError(f"Persona directory not found: {persona_dir}")

    core_file = persona_dir / f"貫通軸_{version}.yaml"
    if not core_file.exists():
        raise FileNotFoundError(
            f"貫通軸_{version}.yaml not found in {persona_dir}"
        )
    core_text = core_file.read_text(encoding="utf-8")

    # 關係層_<counterpart>_<version>.yaml — counterpart is whatever matches
    rel_pattern = re.compile(
        rf"^關係層_(?P<counterpart>.+)_{re.escape(version)}\.yaml$"
    )
    relationship_texts: dict[str, str] = {}
    for rel_file in persona_dir.glob(f"關係層_*_{version}.yaml"):
        match = rel_pattern.match(rel_file.name)
        if match:
            counterpart = match.group("counterpart")
            relationship_texts[counterpart] = rel_file.read_text(encoding="utf-8")

    return Persona(
        name=persona_dir.name,
        version=version,
        core_text=core_text,
        relationship_texts=relationship_texts,
    )

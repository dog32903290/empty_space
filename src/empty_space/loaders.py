"""Loaders for persona / setting / experiment YAML files.

Persona directory layout example:
    persona/六個劇中人/母親/
        貫通軸_v3_tension.yaml         ← core_text
        貫通軸_baseline.yaml
        關係層_兒子_v3_tension.yaml    ← relationship_texts["兒子"]
        關係層_兒子_baseline.yaml
"""
import re
from pathlib import Path

import yaml

from empty_space.paths import PERSONA_ROOT, EXPERIMENTS_DIR
from empty_space.schemas import Persona, Setting, ExperimentConfig
from empty_space.judge import parse_judge_principles, parse_stage_mode_contexts


def _resolve_under(root: Path, rel_path: str) -> Path:
    """Resolve rel_path under root and reject any attempt to escape root.

    Returns the resolved absolute Path. Raises ValueError if rel_path
    walks outside root (e.g., '../../../etc/passwd').
    """
    root_resolved = root.resolve()
    target = (root / rel_path).resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(
            f"rel_path escapes its root: {rel_path!r} resolves to {target}"
        )
    return target


def load_persona(rel_path: str, version: str) -> Persona:
    """Load a Persona from PERSONA_ROOT / rel_path, filtering by version.

    Args:
        rel_path: path relative to PERSONA_ROOT (e.g., "六個劇中人/母親")
        version: version suffix (e.g., "v3_tension", "baseline")

    Returns:
        Persona with core_text and all matching relationship_texts.

    Raises:
        ValueError: if rel_path escapes PERSONA_ROOT.
        FileNotFoundError: if directory or 貫通軸 file missing.
    """
    persona_dir = _resolve_under(PERSONA_ROOT, rel_path)
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

    # Level 4: optional v3 judge files
    judge_principles_text = ""
    jp_file = persona_dir / "judge_principles_v3.yaml"
    if jp_file.exists():
        judge_principles_text = parse_judge_principles(
            jp_file.read_text(encoding="utf-8")
        )

    stage_mode_contexts_parsed: dict[str, dict[str, str]] = {}
    smc_file = persona_dir / "stage_mode_contexts_v3.yaml"
    if smc_file.exists():
        raw = yaml.safe_load(smc_file.read_text(encoding="utf-8")) or {}
        stage_mode_contexts_parsed = parse_stage_mode_contexts(raw)

    return Persona(
        name=persona_dir.name,
        version=version,
        core_text=core_text,
        relationship_texts=relationship_texts,
        judge_principles_text=judge_principles_text,
        stage_mode_contexts_parsed=stage_mode_contexts_parsed,
    )


def load_setting(rel_path: str) -> Setting:
    """Load a Setting from PERSONA_ROOT / rel_path.

    Args:
        rel_path: path relative to PERSONA_ROOT (e.g., "六個劇中人/環境_醫院.yaml")

    Returns:
        Setting with name (filename stem) and raw YAML content.

    Raises:
        ValueError: if rel_path escapes PERSONA_ROOT.
        FileNotFoundError: if file missing.
    """
    setting_file = _resolve_under(PERSONA_ROOT, rel_path)
    if not setting_file.exists():
        raise FileNotFoundError(f"Setting file not found: {setting_file}")
    return Setting(
        name=setting_file.stem,
        content=setting_file.read_text(encoding="utf-8"),
    )


def load_experiment(exp_id: str) -> ExperimentConfig:
    """Load an ExperimentConfig from EXPERIMENTS_DIR / <exp_id>.yaml.

    Args:
        exp_id: experiment identifier, matching filename (without .yaml).

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: if experiments/<exp_id>.yaml doesn't exist.
        pydantic.ValidationError: if YAML content doesn't match schema.
    """
    exp_file = EXPERIMENTS_DIR / f"{exp_id}.yaml"
    if not exp_file.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_file}")
    raw = yaml.safe_load(exp_file.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(raw)

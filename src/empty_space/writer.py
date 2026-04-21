"""Atomic per-turn persistence for Phase 2 runs.

Writes to runs/<exp_id>/<timestamp>/:
  - config.yaml (deep copy of ExperimentConfig)
  - turns/turn_NNN.yaml (one file per turn, atomic rename)
  - conversation.md (append per turn, markdown)
  - conversation.jsonl (append per turn, one JSON object per line)
  - meta.yaml (written once at the end)

Atomicity for yaml files: write to .tmp, os.replace → final.
"""
import json
import os
from pathlib import Path

import yaml

from empty_space.schemas import ExperimentConfig


def init_run(out_dir: Path, config: ExperimentConfig) -> None:
    """Create the run directory skeleton and write config.yaml + conversation init."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "turns").mkdir(exist_ok=True)

    # config.yaml — deep copy via pydantic's model_dump
    config_dump = config.model_dump()
    _atomic_write_yaml(out_dir / "config.yaml", config_dump)

    # conversation.md header
    scene = config.scene_premise or ""
    header_lines = [
        f"# {config.exp_id} @ {out_dir.name}",
        "",
    ]
    if scene:
        header_lines.extend([f"**場景**：{scene.rstrip()}", "", "---", ""])
    else:
        header_lines.extend(["---", ""])
    (out_dir / "conversation.md").write_text(
        "\n".join(header_lines), encoding="utf-8"
    )

    # conversation.jsonl — empty file, appended per turn
    (out_dir / "conversation.jsonl").write_text("", encoding="utf-8")


def _atomic_write_yaml(path: Path, data: object) -> None:
    """Write YAML via .tmp + os.replace for atomicity."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)

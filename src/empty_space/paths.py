"""Path configuration for empty-space.

Persona materials live in sibling repo 演員方法論xhermes/persona/ —
柏為's active persona library. Do not copy or symlink; reference by path.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PERSONA_ROOT = PROJECT_ROOT.parent / "演員方法論xhermes" / "persona"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RUNS_DIR = PROJECT_ROOT / "runs"
LEDGERS_DIR = PROJECT_ROOT / "ledgers"

"""Tests for loader reading v3 judge files (optional, empty when missing)."""
from pathlib import Path

import pytest
import yaml

from empty_space.loaders import load_persona


@pytest.fixture
def persona_with_v3(tmp_path, monkeypatch):
    """Build a minimal persona dir with v3 files."""
    root = tmp_path / "persona_root"
    pdir = root / "test_group" / "母親"
    pdir.mkdir(parents=True)

    (pdir / "貫通軸_v3_tension.yaml").write_text("core text", encoding="utf-8")
    (pdir / "關係層_兒子_v3_tension.yaml").write_text("rel text", encoding="utf-8")
    (pdir / "judge_principles_v3.yaml").write_text(
        "MODE_傾向:\n  收: 0.6\n  放: 0.05\n  在: 0.35\n", encoding="utf-8",
    )
    (pdir / "stage_mode_contexts_v3.yaml").write_text(
        yaml.safe_dump({
            "前置積累_收": {
                "身體": "鯨的下潛", "語言形態": "極短", "張力狀態": "拉力 > 推力",
            },
        }, allow_unicode=True),
        encoding="utf-8",
    )

    monkeypatch.setattr("empty_space.loaders.PERSONA_ROOT", root)
    return root, pdir


def test_load_persona_with_v3_files(persona_with_v3):
    root, _ = persona_with_v3
    p = load_persona("test_group/母親", version="v3_tension")
    assert "MODE_傾向" in p.judge_principles_text
    assert "前置積累_收" in p.stage_mode_contexts_parsed
    assert p.stage_mode_contexts_parsed["前置積累_收"]["身體傾向"] == "鯨的下潛"


def test_load_persona_without_v3_files_has_empty_fields(tmp_path, monkeypatch):
    root = tmp_path / "persona_root"
    pdir = root / "g" / "父親"
    pdir.mkdir(parents=True)
    (pdir / "貫通軸_v3_tension.yaml").write_text("core", encoding="utf-8")
    monkeypatch.setattr("empty_space.loaders.PERSONA_ROOT", root)

    p = load_persona("g/父親", version="v3_tension")
    assert p.judge_principles_text == ""
    assert p.stage_mode_contexts_parsed == {}

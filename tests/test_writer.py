"""Tests for writer — run directory init, per-turn append, meta writeout."""
import json
from pathlib import Path

import pytest
import yaml

from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    Termination,
)
from empty_space.writer import init_run


@pytest.fixture
def sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="test_exp_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="父親在 ICU。",
        initial_state=InitialState(verb="承受", stage="前置積累", mode="基線"),
        director_events={3: "護士推床進來"},
        max_turns=5,
        termination=Termination(),
    )


def test_init_run_creates_directory_structure(tmp_path, sample_config):
    out_dir = tmp_path / "runs" / "test_exp_001" / "2026-04-21T11-30-15"
    init_run(out_dir, sample_config)
    assert out_dir.is_dir()
    assert (out_dir / "turns").is_dir()


def test_init_run_writes_config_yaml_deep_copy(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    config_file = out_dir / "config.yaml"
    assert config_file.is_file()
    loaded = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert loaded["exp_id"] == "test_exp_001"
    assert loaded["scene_premise"] == "父親在 ICU。"
    assert loaded["director_events"] == {3: "護士推床進來"}
    assert loaded["max_turns"] == 5


def test_init_run_initializes_conversation_md_with_scene(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    conv_md = (out_dir / "conversation.md").read_text(encoding="utf-8")
    assert "test_exp_001" in conv_md
    assert "父親在 ICU。" in conv_md


def test_init_run_creates_empty_conversation_jsonl(tmp_path, sample_config):
    out_dir = tmp_path / "run"
    init_run(out_dir, sample_config)
    conv_jsonl = out_dir / "conversation.jsonl"
    assert conv_jsonl.is_file()
    assert conv_jsonl.read_text(encoding="utf-8") == ""

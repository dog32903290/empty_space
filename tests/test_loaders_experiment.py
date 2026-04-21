import pytest
from empty_space.loaders import load_experiment


def test_load_first_experiment():
    config = load_experiment("mother_x_son_hospital_v3_001")
    assert config.exp_id == "mother_x_son_hospital_v3_001"
    assert config.protagonist.path == "六個劇中人/母親"
    assert config.protagonist.version == "v3_tension"
    assert config.counterpart.path == "六個劇中人/兒子"
    assert config.setting.path == "六個劇中人/環境_醫院.yaml"
    assert config.initial_state.verb == "承受（靠近）"
    assert config.scene_premise is not None
    assert "ICU" in config.scene_premise
    assert config.director_events == {}
    assert config.max_turns == 20
    assert config.termination.on_fire_release is True


def test_load_missing_experiment_raises():
    with pytest.raises(FileNotFoundError):
        load_experiment("nonexistent_experiment")

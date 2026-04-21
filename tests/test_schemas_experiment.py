import pytest
from pydantic import ValidationError

from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    Termination,
)


def test_experiment_config_full_construction():
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        scene_premise="他們在醫院。父親在 ICU。",
        initial_state=InitialState(
            verb="承受（靠近）", stage="前置積累", mode="基線"
        ),
        director_events={3: "護士推一張空床進病房"},
        max_turns=20,
        termination=Termination(on_fire_release=True, on_basin_lock=True),
    )
    assert config.exp_id == "mother_x_son_hospital_v3_001"
    assert config.protagonist.version == "v3_tension"
    assert config.scene_premise == "他們在醫院。父親在 ICU。"
    assert config.director_events == {3: "護士推一張空床進病房"}
    assert config.max_turns == 20


def test_termination_has_defaults():
    t = Termination()
    assert t.on_fire_release is True
    assert t.on_basin_lock is True


def test_experiment_defaults_work_with_minimal_input():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.max_turns == 20
    assert config.scene_premise is None
    assert config.director_events == {}
    assert config.termination.on_fire_release is True


def test_experiment_rejects_unknown_fields_not_required():
    # scene_premise is optional; missing is fine
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.scene_premise is None


def test_director_events_accepts_int_keys():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        initial_state=InitialState(verb="v", stage="s", mode="m"),
        director_events={3: "event A", 10: "event B"},
    )
    assert 3 in config.director_events
    assert config.director_events[10] == "event B"

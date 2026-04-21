import pytest
from pydantic import ValidationError

from empty_space.schemas import (
    ExperimentConfig,
    PersonaRef,
    SettingRef,
    InitialState,
    ScriptedTurn,
    Termination,
)


def test_experiment_config_full_construction():
    config = ExperimentConfig(
        exp_id="mother_x_son_hospital_v3_001",
        protagonist=PersonaRef(path="六個劇中人/母親", version="v3_tension"),
        counterpart=PersonaRef(path="六個劇中人/兒子", version="v3_tension"),
        setting=SettingRef(path="六個劇中人/環境_醫院.yaml"),
        protagonist_opener="你是一個中年女人",
        counterpart_system="你是一個年輕男人",
        initial_state=InitialState(
            verb="承受（靠近）", stage="前置積累", mode="基線"
        ),
        scripted_turns={
            8: ScriptedTurn(
                speaker="counterpart",
                content="你從來沒有找過我",
            )
        },
        max_turns=20,
        termination=Termination(on_fire_release=True, on_basin_lock=True),
    )
    assert config.exp_id == "mother_x_son_hospital_v3_001"
    assert config.protagonist.version == "v3_tension"
    assert config.scripted_turns[8].speaker == "counterpart"
    assert config.max_turns == 20


def test_termination_has_defaults():
    t = Termination()
    assert t.on_fire_release is True
    assert t.on_basin_lock is True


def test_scripted_turn_speaker_must_be_valid():
    with pytest.raises(ValidationError):
        ScriptedTurn(speaker="narrator", content="invalid speaker")


def test_experiment_defaults_work_with_minimal_input():
    config = ExperimentConfig(
        exp_id="x",
        protagonist=PersonaRef(path="p", version="v"),
        counterpart=PersonaRef(path="q", version="v"),
        setting=SettingRef(path="s.yaml"),
        protagonist_opener="a",
        counterpart_system="b",
        initial_state=InitialState(verb="v", stage="s", mode="m"),
    )
    assert config.max_turns == 20
    assert config.scripted_turns == {}
    assert config.termination.on_fire_release is True

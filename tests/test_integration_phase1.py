"""Phase 1 integration test: load a real experiment with all its
persona/setting dependencies, verify everything stitches together.
"""
from empty_space.loaders import load_experiment, load_persona, load_setting


def test_full_experiment_dependency_chain():
    config = load_experiment("mother_x_son_hospital_v3_001")
    assert config.exp_id == "mother_x_son_hospital_v3_001"

    protagonist = load_persona(
        config.protagonist.path,
        version=config.protagonist.version,
    )
    assert protagonist.name == "母親"
    assert "兒子" in protagonist.relationship_texts

    counterpart = load_persona(
        config.counterpart.path,
        version=config.counterpart.version,
    )
    assert counterpart.name == "兒子"
    assert "母親" in counterpart.relationship_texts

    setting = load_setting(config.setting.path)
    assert setting.name == "環境_醫院"

    assert "關係語境" in protagonist.relationship_texts["兒子"]
    assert "關係語境" in counterpart.relationship_texts["母親"]

    # Phase 2: scene_premise is present, director_events is empty by default
    assert config.scene_premise is not None
    assert config.director_events == {}

"""Phase 1 integration test: load a real experiment with all its
persona/setting dependencies, verify everything stitches together.
"""
from empty_space.loaders import load_experiment, load_persona, load_setting


def test_full_experiment_dependency_chain():
    # 1. Load experiment config
    config = load_experiment("mother_x_son_hospital_v3_001")
    assert config.exp_id == "mother_x_son_hospital_v3_001"

    # 2. Load protagonist using config refs
    protagonist = load_persona(
        config.protagonist.path,
        version=config.protagonist.version,
    )
    assert protagonist.name == "母親"
    assert "兒子" in protagonist.relationship_texts

    # 3. Load counterpart
    counterpart = load_persona(
        config.counterpart.path,
        version=config.counterpart.version,
    )
    assert counterpart.name == "兒子"
    assert "母親" in counterpart.relationship_texts

    # 4. Load setting
    setting = load_setting(config.setting.path)
    assert setting.name == "環境_醫院"

    # 5. Relationships are bidirectional (both persona have the other's layer)
    assert "關係語境" in protagonist.relationship_texts["兒子"]
    assert "關係語境" in counterpart.relationship_texts["母親"]

    # 6. Scripted turn 8 is present and correctly shaped
    assert 8 in config.scripted_turns
    assert config.scripted_turns[8].speaker == "counterpart"

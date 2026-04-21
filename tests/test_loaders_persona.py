import pytest
from empty_space.loaders import load_persona


def test_load_mother_v3_tension():
    p = load_persona("六個劇中人/母親", version="v3_tension")
    assert p.name == "母親"
    assert p.version == "v3_tension"
    assert "核心既定事實" in p.core_text
    # 兒子 is the only counterpart for 母親 in this repo
    assert "兒子" in p.relationship_texts
    assert "關係語境" in p.relationship_texts["兒子"]


def test_load_mother_baseline_is_prose():
    p = load_persona("六個劇中人/母親_baseline", version="baseline")
    assert p.name == "母親_baseline"
    assert p.version == "baseline"
    # baseline is narrative text
    assert len(p.core_text) > 0


def test_load_director_actor_pair_v3():
    d = load_persona("導演與演員/導演", version="v3_tension")
    assert d.name == "導演"
    assert "演員" in d.relationship_texts

    a = load_persona("導演與演員/演員", version="v3_tension")
    assert a.name == "演員"
    assert "導演" in a.relationship_texts


def test_load_nonexistent_persona_raises():
    with pytest.raises(FileNotFoundError):
        load_persona("六個劇中人/不存在", version="v3_tension")


def test_load_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_persona("六個劇中人/母親", version="nonexistent_version")

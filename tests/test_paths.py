from empty_space.paths import (
    PROJECT_ROOT,
    PERSONA_ROOT,
    EXPERIMENTS_DIR,
    RUNS_DIR,
    LEDGERS_DIR,
)


def test_project_root_is_the_repo():
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_persona_root_points_to_sibling():
    assert PERSONA_ROOT.name == "persona"
    assert PERSONA_ROOT.parent.name == "演員方法論xhermes"
    assert PERSONA_ROOT.is_dir()


def test_persona_root_has_expected_characters():
    assert (PERSONA_ROOT / "六個劇中人" / "母親").is_dir()
    assert (PERSONA_ROOT / "六個劇中人" / "兒子").is_dir()
    assert (PERSONA_ROOT / "導演與演員").is_dir()


def test_local_dirs_under_project_root():
    assert EXPERIMENTS_DIR.parent == PROJECT_ROOT
    assert RUNS_DIR.parent == PROJECT_ROOT
    assert LEDGERS_DIR.parent == PROJECT_ROOT

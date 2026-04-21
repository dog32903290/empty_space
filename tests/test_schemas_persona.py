from empty_space.schemas import Persona


def test_persona_stores_name_version_and_content():
    p = Persona(
        name="母親",
        version="v3_tension",
        core_text="核心既定事實:\n  - 她的孩子被帶走了",
        relationship_texts={"兒子": "關係語境:\n  - 他回來之後看她"},
    )
    assert p.name == "母親"
    assert p.version == "v3_tension"
    assert "孩子被帶走" in p.core_text
    assert "兒子" in p.relationship_texts
    assert "他回來之後" in p.relationship_texts["兒子"]


def test_persona_relationship_texts_defaults_empty():
    p = Persona(name="母親", version="baseline", core_text="prose narrative")
    assert p.relationship_texts == {}

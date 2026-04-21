from empty_space.schemas import Setting


def test_setting_stores_name_and_content():
    s = Setting(
        name="環境_醫院",
        content="既定事實:\n  - 日光燈很白",
    )
    assert s.name == "環境_醫院"
    assert "日光燈" in s.content

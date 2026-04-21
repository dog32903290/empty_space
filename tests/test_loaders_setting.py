import pytest
from empty_space.loaders import load_setting


def test_load_hospital_setting():
    s = load_setting("六個劇中人/環境_醫院.yaml")
    assert s.name == "環境_醫院"
    assert "日光燈" in s.content
    assert "消毒水" in s.content
    assert "情緒動詞" in s.content


def test_load_nonexistent_setting_raises():
    with pytest.raises(FileNotFoundError):
        load_setting("六個劇中人/環境_不存在.yaml")

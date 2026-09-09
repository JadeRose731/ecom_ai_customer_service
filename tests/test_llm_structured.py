# tests/test_llm_structured.py
"""get_chat_model 的 temperature 参数:默认仍 0.3,裁判显式传 0;streaming 组合不丢。"""
from app.core.llm import get_chat_model


def test_default_temperature_unchanged():
    assert get_chat_model().temperature == 0.3


def test_judge_temperature_zero():
    assert get_chat_model(temperature=0).temperature == 0


def test_streaming_combines_with_temperature():
    m = get_chat_model(streaming=True, temperature=0)
    assert m.streaming is True and m.temperature == 0

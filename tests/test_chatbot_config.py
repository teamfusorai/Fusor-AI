"""Unit tests for chatbot_config build_smart_system_prompt."""

from chatbot_config import build_smart_system_prompt


def test_build_smart_system_prompt_default():
    out = build_smart_system_prompt()
    assert "retrieval-augmented" in out or "retrieved context" in out
    assert "TONE:" in out
    assert "Never hallucinate" in out or "hallucinate" in out.lower()


def test_build_smart_system_prompt_with_name():
    out = build_smart_system_prompt(chatbot_name="SupportBot")
    assert "SupportBot" in out
    assert "CHATBOT IDENTITY" in out or "You are" in out


def test_build_smart_system_prompt_with_tone():
    out = build_smart_system_prompt(tone="professional")
    assert "professional" in out.lower()
    out2 = build_smart_system_prompt(tone="technical")
    assert "technical" in out2.lower()


def test_build_smart_system_prompt_with_industry():
    out = build_smart_system_prompt(industry="healthcare")
    assert "healthcare" in out.lower()
    assert "industry" in out.lower()


def test_build_smart_system_prompt_custom_base():
    custom = "You are a helpful assistant."
    out = build_smart_system_prompt(custom_system_prompt=custom)
    assert out.startswith(custom) or custom in out
    assert "TONE:" in out

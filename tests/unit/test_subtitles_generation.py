"""Focused tests for subtitle generation fallbacks."""

from modules.core import utils


def _build_srt_result(text: str, start: float = 0.0, end: float = 1.0) -> dict[str, object]:
    return {"text": text, "segments": [{"start": start, "end": end, "text": text}]}


def test_generate_srt_standard_case() -> None:
    """Test SRT generation for a standard segment payload."""
    srt = utils.generate_srt(_build_srt_result("Hello world"))
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "Hello world" in srt


def test_generate_srt_without_segments_uses_text_fallback() -> None:
    """Test SRT generation when only top-level text is provided."""
    srt = utils.generate_srt({"text": "Simple text"})
    assert "Simple text" in srt
    assert "00:00:00,000 --> 00:00:05,000" in srt


def test_generate_srt_empty_result_uses_placeholder() -> None:
    """Test SRT generation for an empty result payload."""
    srt = utils.generate_srt({})
    assert "[No dialogue detected]" in srt

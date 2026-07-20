"""Tests for subtitle promo generation utilities"""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

from modules.core import utils


@contextmanager
def _enabled_promo_patches() -> Iterator[None]:
    with (
        mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", True),
        mock.patch("modules.core.config.SUBTITLE_PROMO_TEXT", "Made with Whisper Pro ASR"),
        mock.patch("modules.core.config.SUBTITLE_PROMO_DURATION", 3.0),
    ):
        yield


def test_subtitle_promo_generation_enabled() -> None:
    """Promo cards should be prepended when promo output is enabled."""
    res = {"segments": [{"start": 1.0, "end": 2.5, "text": "Dialogue text"}]}

    with _enabled_promo_patches():
        srt = utils.generate_srt(res)
        assert "1\n00:00:00,000 --> 00:00:03,000\nMade with Whisper Pro ASR" in srt
        assert "2\n00:00:01,000 --> 00:00:02,500\nDialogue text" in srt

        vtt = utils.generate_vtt(res)
        assert "1\n00:00:00.000 --> 00:00:03.000\nMade with Whisper Pro ASR" in vtt
        assert "2\n00:00:01.000 --> 00:00:02.500\nDialogue text" in vtt


def test_subtitle_promo_generation_disabled() -> None:
    """Promo cards should be omitted when promo output is disabled."""
    res = {"segments": [{"start": 1.0, "end": 2.5, "text": "Dialogue text"}]}

    with mock.patch("modules.core.config.SUBTITLE_PROMO_ENABLED", False):
        srt = utils.generate_srt(res)
        assert "Made with Whisper Pro ASR" not in srt
        assert "1\n00:00:01,000 --> 00:00:02,500\nDialogue text" in srt

        vtt = utils.generate_vtt(res)
        assert "Made with Whisper Pro ASR" not in vtt
        assert "1\n00:00:01.000 --> 00:00:02.500\nDialogue text" in vtt


def test_subtitle_promo_generation_empty_result_shifts_timing() -> None:
    """Promo insertion should shift the fallback block when no segments exist."""
    empty_res = {"segments": []}

    with _enabled_promo_patches():
        srt = utils.generate_srt(empty_res)
        assert "1\n00:00:00,000 --> 00:00:03,000\nMade with Whisper Pro ASR" in srt
        assert "2\n00:00:03,000 --> 00:00:08,000\n[No dialogue detected]" in srt

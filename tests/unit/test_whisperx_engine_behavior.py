"""Additional coverage tests for modules/inference/engines/whisperx_engine.py."""

from unittest import mock

from modules.inference.engines import whisperx_engine


def test_unsupported_whisperx_options_collects_all_flags():
    """Unsupported option helper should report all non-supported flags."""
    unsupported = getattr(whisperx_engine, "_unsupported_whisperx_options")(
        initial_prompt="ctx",
        vad_filter=False,
        word_timestamps=True,
    )
    assert unsupported == ["initial_prompt", "vad_filter", "word_timestamps"]


def test_try_detect_language_candidate_logs_and_returns_none_on_exception(caplog):
    """Language detection candidate failures should log warning and return None."""
    caplog.set_level("WARNING")
    candidate = mock.MagicMock()
    candidate.detect_language.side_effect = RuntimeError("boom")

    result = getattr(whisperx_engine, "_try_detect_language_candidate")("model", candidate, "audio")

    assert result is None
    assert "using fallback" in caplog.text


def test_detect_language_via_transcribe_fallback_defaults_to_en():
    """Fallback detection should default to English when language is missing."""
    model = mock.MagicMock()
    model.transcribe.return_value = {}

    lang, prob, probs = getattr(whisperx_engine, "_detect_language_via_transcribe_fallback")(model, "audio")

    assert (lang, prob, probs) == ("en", 1.0, [("en", 1.0)])


def test_whisperx_engine_transcribe_logs_unsupported_options(caplog):
    """Engine transcribe should log ignored unsupported options and still return segments."""
    caplog.set_level("WARNING")
    mock_whisperx = mock.MagicMock()
    mock_whisperx.load_audio.return_value = "audio-array"
    mock_whisperx.load_model.return_value.transcribe.return_value = {
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
    }

    with mock.patch("importlib.import_module", return_value=mock_whisperx):
        engine = whisperx_engine.WhisperXEngine(model_id="m", device="cpu")
        segments, info = engine.transcribe(
            "x.wav",
            initial_prompt="ctx",
            vad_filter=False,
            word_timestamps=True,
        )

    assert len(list(segments)) == 1
    assert info.language == "en"
    assert "Ignoring unsupported options" in caplog.text

"""Tests for new ASR parameters, idle timeout, and subtitle wrapping improvements."""
# pylint: disable=protected-access, redefined-outer-name
import time
from unittest import mock
import pytest
from flask import Flask

from modules.inference import model_manager, scheduler
from modules import utils, config
from modules.api import routes_asr


@pytest.fixture(autouse=True)
def reset_state():
    """Reset model_manager pools and scheduler states between tests."""
    model_manager._MODEL_POOL.clear()
    model_manager._PREPROCESSOR_POOL.clear()
    model_manager._DIARIZE_POOL.clear()
    model_manager._ALIGN_POOL.clear()

    with mock.patch("modules.config.HARDWARE_UNITS", [{"id": "CPU", "type": "CPU", "name": "CPU"}]):
        from modules.inference.scheduler import SchedulerState
        scheduler.STATE = SchedulerState()
        scheduler.STATE.engine_initialized = True

    # Reset thread context and idle monitor state
    utils.THREAD_CONTEXT.is_priority = False
    if hasattr(utils.THREAD_CONTEXT, 'assigned_unit'):
        utils.THREAD_CONTEXT.assigned_unit = None

    model_manager._MONITOR_THREAD_STARTED = False
    yield


def test_routes_extract_new_params():
    """Verify that ASR routes correctly parse new parameters."""
    app = Flask(__name__)
    app.register_blueprint(routes_asr.bp)

    # Test full parameters extraction
    with app.test_request_context(
        '/asr?initial_prompt=testprompt&vad_filter=false&word_timestamps=true&max_line_width=40&max_line_count=2'
    ):
        params = routes_asr._get_request_params()
        assert params['initial_prompt'] == 'testprompt'
        assert params['vad_filter'] is False
        assert params['word_timestamps'] is True
        assert params['max_line_width'] == 40
        assert params['max_line_count'] == 2

    # Test default values when omitted
    with app.test_request_context('/asr'):
        params = routes_asr._get_request_params()
        assert params['initial_prompt'] is None
        assert params['vad_filter'] is True
        assert params['word_timestamps'] is False
        assert params['max_line_width'] is None
        assert params['max_line_count'] is None

    # Test malformed width/count integers fallback to None
    with app.test_request_context('/asr?max_line_width=invalid&max_line_count=invalid'):
        params = routes_asr._get_request_params()
        assert params['max_line_width'] is None
        assert params['max_line_count'] is None


def test_model_manager_forwards_new_params():
    """Verify that model_manager.run_transcription forwards new parameters to transcribe call."""
    mock_model = mock.MagicMock()
    mock_info = mock.MagicMock(language="en", language_probability=0.95, duration=5.0)
    mock_segment = mock.MagicMock(start=0.0, end=1.0, text="hello")
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    model_manager._MODEL_POOL["CPU"] = mock_model

    model_manager.run_transcription(
        "test.wav",
        language="en",
        task="transcribe",
        initial_prompt="hello test",
        vad_filter=False,
        word_timestamps=True
    )

    mock_model.transcribe.assert_called_once_with(
        "test.wav",
        language="en",
        task="transcribe",
        beam_size=config.DEFAULT_BEAM_SIZE,
        initial_prompt="hello test",
        vad_filter=False,
        vad_parameters={"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
        word_timestamps=True
    )


def test_model_idle_timeout_reclamation():
    """Verify that the background idle timeout thread successfully offloads models."""
    pm = mock.MagicMock()
    model_manager._PREPROCESSOR_POOL["CPU"] = pm
    model_manager._MODEL_POOL["CPU"] = mock.MagicMock()
    scheduler.STATE.active_sessions = 1

    # Configure timeout of 1 second
    with mock.patch("modules.config.MODEL_IDLE_TIMEOUT", 1), \
            mock.patch("modules.inference.model_manager.utils.get_system_telemetry", return_value={}):

        # Simulates task registration/completion lifecycle
        model_manager.decrement_active_session()
        assert scheduler.STATE.active_sessions == 0

        # Model should still be in pool initially
        assert len(model_manager._MODEL_POOL) == 1

        # Sleep to allow idle timeout monitor thread to run and trigger offload
        time.sleep(6)

        # Monitor thread should have cleared the pools
        assert len(model_manager._MODEL_POOL) == 0
        pm.unload_model.assert_called_once()


def test_subtitle_wrapping_logic():
    """Verify text wrapping utilities and layout constraints in SRT/VTT writers."""
    # Test _wrap_text directly
    long_text = "This is a very long text that we want to wrap to a maximum width of characters."
    wrapped = utils._wrap_text(long_text, max_line_width=20)
    lines = wrapped.split("\n")
    for line in lines:
        assert len(line) <= 20

    # Test max_line_count limit
    limited = utils._wrap_text(long_text, max_line_width=20, max_line_count=2)
    assert len(limited.split("\n")) == 2

    # Test SRT/VTT formatters wrapping
    result = {
        "segments": [
            {"start": 0.0, "end": 5.0, "text": long_text}
        ]
    }

    srt_out = utils.generate_srt(result, max_line_width=20, max_line_count=2)
    # The segment text block should contain wrapped lines and be capped at 2 lines
    assert "wrap to a maximum" not in srt_out
    assert "This is a very long\ntext that we want to" in srt_out

    vtt_out = utils.generate_vtt(result, max_line_width=20, max_line_count=2)
    assert "wrap to a maximum" not in vtt_out
    assert "This is a very long\ntext that we want to" in vtt_out

"""Unit tests for runtime segment processing helpers."""

from types import SimpleNamespace
from unittest import mock

from modules.inference.runtime import model_segment_processing


def test_consume_transcription_segments_updates_metadata_and_progress():
    """Consumption should emit SRT metadata, progress updates, and segment dicts."""
    segments = [
        SimpleNamespace(start=0.0, end=1.5, text=" hello ", words=[SimpleNamespace(start=0.0, end=0.5, word="he", probability=0.9)]),
        SimpleNamespace(start=1.5, end=3.0, text=" world ", words=None),
    ]
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", side_effect=["A", "B"]),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata") as mock_metadata,
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
        mock.patch("modules.inference.runtime.model_segment_processing.logger.info") as mock_log,
    ):
        results = model_segment_processing.consume_transcription_segments(
            segments,
            info,
            "translate",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {"start": 0.0, "end": 1.5, "text": "hello", "words": [{"start": 0.0, "end": 0.5, "word": "he", "probability": 0.9}]},
        {"start": 1.5, "end": 3.0, "text": "world"},
    ]
    assert mock_metadata.call_count == 2
    assert mock_progress.call_count == 2
    assert mock_log.call_count == 1


def test_consume_transcription_segments_logs_transcribing_progress():
    """Transcribe task should use the transcribing verb in progress updates."""
    segment = SimpleNamespace(start=0.0, end=2.0, text=" hello ", words=None)
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
    ):
        model_segment_processing.consume_transcription_segments(
            [segment],
            info,
            "transcribe",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert mock_progress.call_args.args[1].startswith("Transcribing")


def test_consume_transcription_segments_skips_progress_when_duration_is_zero():
    """A zero-duration info object should skip segment progress updates."""
    segments = [SimpleNamespace(start=0.0, end=1.0, text=" test ", words=None)]
    info = SimpleNamespace(duration=0.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_metadata") as mock_metadata,
        mock.patch("modules.inference.runtime.model_segment_processing.scheduler.update_task_progress") as mock_progress,
    ):
        results = model_segment_processing.consume_transcription_segments(
            segments,
            info,
            "transcribe",
            diarize=False,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [{"start": 0.0, "end": 1.0, "text": "test"}]
    mock_metadata.assert_called_once()
    mock_progress.assert_not_called()


def test_run_diarization_safe_falls_back_to_raw_segments():
    """Diarization failures should preserve raw segment content and words."""
    info = SimpleNamespace(duration=10.0, language="en")

    with mock.patch(
        "modules.inference.runtime.model_segment_processing.diarization.run_diarization",
        side_effect=RuntimeError("boom"),
    ):
        results = model_segment_processing.consume_transcription_segments(
            [SimpleNamespace(start=0.0, end=1.0, text=" hello ", words=[SimpleNamespace(start=0.0, end=0.5, word="hello")])],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 1.0}],
        },
    ]


def test_run_diarization_safe_falls_back_with_words_preserved():
    """Diarization failures should preserve raw segment content and words."""
    segment = SimpleNamespace(start=0.0, end=1.0, text=" hello ", words=[SimpleNamespace(start=0.0, end=0.5, word="hello")])
    info = SimpleNamespace(duration=10.0, language="en")

    with (
        mock.patch("modules.inference.runtime.model_segment_processing.utils.format_single_srt_block", return_value="block"),
        mock.patch(
            "modules.inference.runtime.model_segment_processing.diarization.run_diarization",
            side_effect=RuntimeError("boom"),
        ),
    ):
        results = model_segment_processing.consume_transcription_segments(
            [segment],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "probability": 1.0}],
        },
    ]


def test_run_diarization_safe_returns_empty_list_without_segments():
    """Empty raw segment input should short-circuit before diarization runs."""
    info = SimpleNamespace(language="en")

    with mock.patch("modules.inference.runtime.model_segment_processing.diarization.run_diarization") as mock_run:
        results = model_segment_processing.consume_transcription_segments(
            [],
            info,
            "transcribe",
            diarize=True,
            min_speakers=None,
            max_speakers=None,
            hf_token=None,
            unit_id="CPU",
            processed_path="audio.wav",
            preemption_check=lambda: None,
        )

    assert results == []
    mock_run.assert_not_called()

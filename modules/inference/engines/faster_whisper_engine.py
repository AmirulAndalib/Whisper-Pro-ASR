"""CTranslate2 faster-whisper engine wrapper."""

import importlib
from typing import Any, Optional

from modules.core import config
from modules.inference.engines.base import BaseASREngine


class FasterWhisperEngine(BaseASREngine):
    """CTranslate2 faster-whisper engine."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str,
        device_index: int = 0,
        compute_type: str = "int8",
        cpu_threads: int = 4,
        download_root: Optional[str] = None,
    ):
        faster_whisper = importlib.import_module("faster_whisper")
        self.model = faster_whisper.WhisperModel(
            model_id,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=download_root,
        )

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        word_timestamps: bool = False,
        **kwargs: Any,
    ):
        params = {
            "beam_size": config.DEFAULT_BEAM_SIZE,
            "initial_prompt": initial_prompt,
            "vad_filter": vad_filter,
            "vad_parameters": {"min_silence_duration_ms": config.VAD_MIN_SILENCE_DURATION_MS},
            "word_timestamps": word_timestamps,
        }
        params.update(kwargs)
        return self.model.transcribe(audio_path, language=language, task=task, **params)

    def detect_language(self, audio: Any):
        """Identify the language of audio data without full transcription."""
        return self.model.detect_language(audio)

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model

"""WhisperX engine wrapper."""

import importlib
import logging
from typing import Any, Optional

from modules.core import config
from modules.inference.engines.base import BaseASREngine, build_inference_info, iter_segment_wrappers

logger = logging.getLogger(__name__)


class WhisperXEngine(BaseASREngine):
    """WhisperX engine supporting batch inference."""

    def __init__(self, model_id: str, device: str, compute_type: str = "int8"):
        self.whisperx = importlib.import_module("whisperx")
        self.device = device
        self.compute_type = compute_type
        self.model = self.whisperx.load_model(model_id, device=device, compute_type=compute_type)

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
        unsupported_opts = _unsupported_whisperx_options(initial_prompt, vad_filter, word_timestamps)
        if unsupported_opts:
            logger.warning("[WhisperX] Ignoring unsupported options: %s", ", ".join(unsupported_opts))

        audio = self.whisperx.load_audio(audio_path)
        batch_size = kwargs.get("batch_size", config.DEFAULT_BATCH_SIZE)
        result = self.model.transcribe(audio, batch_size=batch_size, language=language, task=task)
        return iter_segment_wrappers(result), build_inference_info(result, audio_path, language)

    def detect_language(self, audio: Any):
        """Identify language with WhisperX/faster-whisper backend when available."""
        if isinstance(audio, str):
            audio = self.whisperx.load_audio(audio)

        direct_result = _try_detect_language_from_whisperx_model(self.model, audio)
        if direct_result is not None:
            return direct_result
        return _detect_language_via_transcribe_fallback(self.model, audio)

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model


def _unsupported_whisperx_options(initial_prompt: Optional[str], vad_filter: bool, word_timestamps: bool) -> list[str]:
    unsupported = []
    if initial_prompt:
        unsupported.append("initial_prompt")
    if not vad_filter:
        unsupported.append("vad_filter")
    if word_timestamps:
        unsupported.append("word_timestamps")
    return unsupported


def _try_detect_language_from_whisperx_model(model, audio: Any):
    for label, candidate in _language_detection_candidates(model):
        detected = _try_detect_language_candidate(label, candidate, audio)
        if detected is not None:
            return detected
    return None


def _language_detection_candidates(model):
    return [
        ("model.model", getattr(model, "model", None)),
        ("model", model),
    ]


def _try_detect_language_candidate(label: str, candidate, audio: Any):
    if not candidate or not hasattr(candidate, "detect_language"):
        return None
    try:
        return _normalize_language_detection_result(candidate.detect_language(audio))
    except (RuntimeError, ValueError, TypeError, AttributeError, KeyError) as exc:
        logger.warning("[EngineFactory] detect_language(%s) failed, using fallback: %s", label, exc)
        return None


def _normalize_language_detection_result(result):
    lang_code, lang_prob, all_probs_list = result
    return lang_code, float(lang_prob), [(k, float(v)) for k, v in all_probs_list]


def _detect_language_via_transcribe_fallback(model, audio: Any):
    result = model.transcribe(audio, batch_size=1, task="transcribe")
    detected_lang = result.get("language", "en")
    return detected_lang, 1.0, [(detected_lang, 1.0)]

"""Shared engine interfaces and result containers."""

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from modules.core import utils


@dataclass
class InferenceInfo:
    """Standardized info structure returned by engines."""

    language: str
    language_probability: float
    duration: float
    all_language_probs: Optional[list[tuple[str, float]]] = None


@dataclass
class SegmentWrapper:
    """Standardized segment structure yielded by engines."""

    start: float
    end: float
    text: str
    words: Optional[list[Any]] = None


class BaseASREngine:
    """Base interface for all ASR engines."""

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
    ) -> tuple[Iterator[Any], Any]:
        """Run transcription and return (segments_iterator, info_object)."""
        raise NotImplementedError()

    def detect_language(self, audio: Any) -> tuple[str, float, list[tuple[str, float]]]:
        """Identify the language of audio data without full transcription."""
        raise NotImplementedError()

    def unload(self) -> None:
        """Release underlying model resources from memory."""


def build_inference_info(result: dict[str, Any], audio_path: str, language: Optional[str]) -> InferenceInfo:
    """Create a normalized InferenceInfo object from a whisper-style result payload."""
    detected_lang = result.get("language", language or "en")
    duration = utils.get_audio_duration(audio_path)
    return InferenceInfo(
        language=detected_lang,
        language_probability=1.0,
        duration=duration,
        all_language_probs=[(detected_lang, 1.0)],
    )


def iter_segment_wrappers(result: dict[str, Any]) -> Iterator[SegmentWrapper]:
    """Yield normalized SegmentWrapper objects from whisper-style segment dictionaries."""
    for seg in result.get("segments", []):
        yield SegmentWrapper(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", ""),
            words=seg.get("words"),
        )

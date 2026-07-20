"""
Speaker Diarization and Alignment Module using WhisperX.

Known Limitation — Very Long Files (15 h+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WhisperX's ``load_audio()``, ``align()``, and ``DiarizationPipeline()``
load the *entire* processed audio file into RAM as float32 numpy arrays.
At 16 kHz mono the memory cost is approximately:

    duration_sec × 16 000 samples/sec × 4 bytes/sample
    ≈ 3.5 GB for 15 hours
    ≈ 5.5 GB for 24 hours

On top of the raw audio buffer, the alignment model and the diarization
pipeline hold their own state, so peak process RSS during diarization of
a 15-hour file can exceed **8–10 GB**.

If your deployment target cannot accommodate this, either:
  • Disable diarization for long files on the client side (``diarize=false``),
  • Increase the container/host memory accordingly, or
  • Set ``MAX_DIARIZE_DURATION_SEC`` to restrict diarization to shorter files.
"""

import importlib
import logging
import os
from typing import Any

from modules.core import config, utils
from modules.inference import scheduler

logger = logging.getLogger(__name__)

# Caching Pools
ALIGN_POOL = {}
DIARIZE_POOL = {}

# Duration threshold (seconds) above which a RAM warning is emitted before
# attempting diarization.  Set to 0 to disable the warning.  Set the env var
# ``MAX_DIARIZE_DURATION_SEC`` to a positive value to *skip* diarization
# entirely for files longer than this (returns raw segments without speakers).
_DIARIZE_WARN_THRESHOLD_SEC = 14400  # 4 hours
MAX_DIARIZE_DURATION_SEC = int(os.environ.get("MAX_DIARIZE_DURATION_SEC", 0))


def _get_whisperx_device(unit_id: str) -> str:
    """Resolve the WhisperX device (cuda or cpu) based on the unit ID."""
    unit = next((u for u in config.HARDWARE_UNITS if u["id"] == unit_id), None)
    unit_type = unit["type"] if unit else "CPU"
    return "cuda" if unit_type == "CUDA" else "cpu"


def _get_align_model(whisperx: Any, lang_code: str, device: str, unit_id: str) -> tuple[Any, Any]:
    """Load or retrieve the alignment model from the cache pool."""
    align_key = (unit_id, lang_code)
    if align_key not in ALIGN_POOL:
        logger.info("[Diarization] Loading alignment model for language: %s on %s", lang_code, device)
        ALIGN_POOL[align_key] = whisperx.load_align_model(language_code=lang_code, device=device)
    return ALIGN_POOL[align_key]


def _get_diarize_pipeline(whisperx: Any, token: str, device: str, unit_id: str) -> Any:
    """Load or retrieve the diarization pipeline from the cache pool."""
    if unit_id not in DIARIZE_POOL:
        scheduler.update_task_progress(90, "Loading Diarization Model")
        logger.info("[Diarization] Loading diarization pipeline on %s...", device)
        DIARIZE_POOL[unit_id] = whisperx.diarization.DiarizationPipeline(use_auth_token=token, device=device)
    return DIARIZE_POOL[unit_id]


def _format_diarized_segments(alignment_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Format diarized segments back to the standard results format."""
    results = []
    for seg in alignment_result["segments"]:
        seg_dict = {
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker"),
        }
        if "words" in seg:
            seg_dict["words"] = seg["words"]
        results.append(seg_dict)
    return results


def run_diarization(
    *,
    processed_path: str,
    raw_segments: list[dict[str, Any]],
    info: Any,
    language: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
    hf_token: str | None,
    unit_id: str,
) -> list[dict[str, Any]]:
    """Aligns segments and performs speaker diarization using whisperx.

    .. warning::

       For very long files (15 h+) this function will consume several GB of
       RAM because WhisperX loads the full audio into memory.  See module
       docstring for details and mitigation options.
    """
    audio_duration = getattr(info, "duration", 0) or 0

    skip_result = _maybe_skip_diarization_for_duration(audio_duration, raw_segments)
    if skip_result is not None:
        return skip_result

    resolved_hf_token = _resolve_hf_token(hf_token)
    if not resolved_hf_token:
        logger.warning("[Diarization] No Hugging Face token available; returning raw segments without speaker labels.")
        return _format_raw_segments_without_speakers(raw_segments)

    try:
        _warn_if_long_diarization(audio_duration)
        whisperx_device, whisperx = _resolve_whisperx_runtime(unit_id)
        alignment_result, audio = _run_alignment_step(
            whisperx,
            whisperx_device,
            unit_id,
            info=info,
            language=language,
            processed_path=processed_path,
            raw_segments=raw_segments,
        )
        diarize_pipeline = _get_diarize_pipeline(whisperx, resolved_hf_token, whisperx_device, unit_id)
        diarize_segments = _run_diarization_step(diarize_pipeline, audio, min_speakers, max_speakers)
        alignment_result = _assign_speakers_step(whisperx, diarize_segments, alignment_result)
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, KeyError, TypeError) as exc:
        logger.warning("[Diarization] Falling back to raw segments without speaker labels: %s", exc)
        return _format_raw_segments_without_speakers(raw_segments)
    return _format_diarized_segments(alignment_result)


def _maybe_skip_diarization_for_duration(audio_duration: float, raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not 0 < MAX_DIARIZE_DURATION_SEC < audio_duration:
        return None
    estimated_gb = _estimate_audio_ram_gb(audio_duration)
    logger.warning(
        "[Diarization] Skipping — audio duration (%s) exceeds MAX_DIARIZE_DURATION_SEC (%ds). "
        "WhisperX alignment would require ~%.1f GB RAM. Returning raw segments without speaker labels.",
        utils.format_duration(audio_duration),
        MAX_DIARIZE_DURATION_SEC,
        estimated_gb,
    )
    return _format_raw_segments_without_speakers(raw_segments)


def _warn_if_long_diarization(audio_duration: float) -> None:
    if audio_duration <= _DIARIZE_WARN_THRESHOLD_SEC:
        return
    logger.warning(
        "[Diarization] Long file detected (%s). WhisperX will load the full audio as float32 (~%.1f GB). "
        "Ensure sufficient RAM is available.",
        utils.format_duration(audio_duration),
        _estimate_audio_ram_gb(audio_duration),
    )


def _estimate_audio_ram_gb(audio_duration: float) -> float:
    return (audio_duration * 16000 * 4) / (1024**3)


def _format_raw_segments_without_speakers(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            **({"words": seg["words"]} if "words" in seg else {}),
        }
        for seg in raw_segments
    ]


def _resolve_whisperx_runtime(unit_id: str) -> tuple[str, Any]:
    return _get_whisperx_device(unit_id), importlib.import_module("whisperx")


def _run_alignment_step(
    whisperx: Any,
    whisperx_device: str,
    unit_id: str,
    *,
    info: Any,
    language: str | None,
    processed_path: str,
    raw_segments: list[dict[str, Any]],
) -> tuple[dict[str, Any], Any]:
    scheduler.update_task_progress(83, "Loading Alignment Model")
    model_a, metadata = _get_align_model(whisperx, info.language or language or "en", whisperx_device, unit_id)
    scheduler.update_task_progress(85, "Aligning Transcription")
    logger.info("[Diarization] Aligning segments...")
    audio = whisperx.load_audio(processed_path)
    alignment_result = whisperx.align(raw_segments, model_a, metadata, audio, device=whisperx_device, return_char_alignments=False)
    return alignment_result, audio


def _resolve_hf_token(hf_token: str | None) -> str | None:
    token = hf_token or config.DIARIZATION_HF_TOKEN
    return token or None


def _run_diarization_step(
    diarize_pipeline: Any,
    audio: Any,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Any:
    scheduler.update_task_progress(93, "Diarizing Speakers")
    logger.info("[Diarization] Running speaker diarization...")
    return diarize_pipeline(audio, min_speakers=min_speakers, max_speakers=max_speakers)


def _assign_speakers_step(whisperx: Any, diarize_segments: Any, alignment_result: dict[str, Any]) -> dict[str, Any]:
    scheduler.update_task_progress(97, "Assigning Speakers")
    logger.info("[Diarization] Assigning speakers to segments...")
    return whisperx.assign_word_speakers(diarize_segments, alignment_result)

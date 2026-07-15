"""
Core Single-Segment Language Detection Logic.
"""

import gc
import logging
import sys
import time

from modules.core import config
from modules.inference import scheduler
from modules.inference.pipeline import vad

logger = logging.getLogger(__name__)


def run_language_detection(audio_path):
    """Optimized language detection using the faster detect_language API."""
    model_manager = sys.modules["modules.inference.runtime.model_manager"]
    start_time = time.time()
    with model_manager.model_lock_ctx() as (model, _):
        scheduler.update_task_progress(5, "Detection")
        res = model_manager.run_language_detection_core(model, audio_path)
        res["performance"] = {"inference_sec": round(time.time() - start_time, 2)}
        res["segments_processed"] = 1
        scheduler.update_task_metadata(result=res)
        return res


def run_batch_language_detection(audio_path, segment_count):
    """High-performance multi-segment identification scan."""
    model_manager = sys.modules["modules.inference.runtime.model_manager"]
    with model_manager.model_lock_ctx() as (model, _):
        return model_manager.run_batch_language_detection_direct(model, audio_path, segment_count)


def run_batch_language_detection_direct(model, audio_path, segment_count):
    """Direct batch detection without re-acquiring the lock."""
    model_manager = sys.modules["modules.inference.runtime.model_manager"]
    full_audio = None
    try:
        full_audio = vad.decode_audio(audio_path)
        return _detect_segments(model, model_manager, full_audio, segment_count)
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, KeyError, TypeError) as e:
        logger.error("[Engine] Batch detection failed: %s", e)
        return []
    finally:
        _cleanup_batch_detection(full_audio)


def _detect_segments(model, model_manager, full_audio, segment_count) -> list:
    results = []
    segment_len = int(30 * 16000)
    for i in range(segment_count):
        start = i * segment_len
        if start >= len(full_audio):
            break
        end = min(start + segment_len, len(full_audio))
        chunk = full_audio[start:end].copy()
        results.append(model_manager.run_language_detection_core(model, chunk, skip_vad=False))

        # Granular progress for voting (Maps 60% -> 95%)
        progress = 60 + int(((i + 1) / segment_count) * 35)
        stage = f"Inference ({i + 1}/{segment_count} segments)"
        logger.info("[Engine] %s...", stage)
        scheduler.update_task_progress(progress, stage)
    return results


def _cleanup_batch_detection(full_audio):
    if full_audio is not None:
        del full_audio
    gc.collect()


def run_language_detection_core(model, audio_input, skip_vad=False):
    """Internal core using detect_language optimization."""
    speech_sec, no_speech_result = _resolve_speech_duration(audio_input, skip_vad)
    if no_speech_result is not None:
        return no_speech_result
    try:
        audio_input = _sanitize_ld_audio_input(audio_input)
    except (ImportError, RuntimeError, OSError, ValueError, AttributeError, TypeError) as sanitize_err:
        logger.info("[Engine] Audio sanitize fallback: %s", sanitize_err)
    try:
        return _detect_language_primary(model, audio_input, speech_sec)
    except tuple([Exception]) as e:
        return _detect_language_fallback(model, audio_input, speech_sec, e)


def _resolve_speech_duration(audio_input, skip_vad: bool) -> tuple[float, dict | None]:
    if skip_vad:
        return 30.0, None
    speech_ts = _get_ld_speech_ts(audio_input)
    if not speech_ts:
        return 0.0, _no_speech_detection_result()
    speech_sec = sum(ts["end"] - ts["start"] for ts in speech_ts)
    return speech_sec, None


def _no_speech_detection_result() -> dict:
    return {
        "detected_language": "en",
        "language": "en",
        "confidence": 0.0,
        "all_probabilities": {"en": 0.0},
        "speech_duration": 0.0,
    }


def _detect_language_primary(model, audio_input, speech_sec: float) -> dict:
    lang_code, lang_prob, all_probs_list = model.detect_language(audio_input)
    logger.info("[Engine] Identified: %s (%.1f%%)", lang_code, lang_prob * 100)
    all_probs = dict(all_probs_list) if all_probs_list else {lang_code: lang_prob}
    return {
        "detected_language": lang_code,
        "language": lang_code,
        "confidence": lang_prob,
        "all_probabilities": {k: v for k, v in all_probs.items() if v >= 0.001},
        "speech_duration": round(speech_sec, 3),
    }


def _get_ld_speech_ts(audio_input) -> list:
    if isinstance(audio_input, str):
        return vad.get_speech_timestamps_from_path(audio_input, threshold=config.LD_VAD_THRESHOLD)
    return vad.get_speech_timestamps(audio_input, threshold=config.LD_VAD_THRESHOLD)


def _sanitize_ld_audio_input(audio_input):
    if isinstance(audio_input, str):
        audio_input = vad.decode_audio(audio_input)
    if hasattr(audio_input, "astype"):
        audio_input = audio_input.astype("float32")
    return audio_input


def _detect_language_fallback(model, audio_input, speech_sec, e) -> dict:
    logger.info("[Engine] detect_language fallback: %s", e)
    _, info = model.transcribe(audio_input, beam_size=1, task="transcribe")
    all_probs = dict(info.all_language_probs) if info.all_language_probs else {}
    return {
        "detected_language": info.language,
        "language": info.language,
        "confidence": info.language_probability,
        "all_probabilities": {k: v for k, v in all_probs.items() if v >= 0.001},
        "speech_duration": round(speech_sec, 3),
    }

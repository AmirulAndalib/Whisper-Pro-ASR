"""
Post-processing filters for transcription outputs.
"""

import logging

from modules.core import config

logger = logging.getLogger(__name__)


def post_process_results(result, _audio_path=None):
    """Applies quality filters to the raw transcription output."""
    segments = _extract_segments_or_none(result)
    if segments is None:
        return result

    processed_segments = []
    repetition_count = 0
    last_text = ""

    for seg in segments:
        text = seg.get("text", "").strip()
        repetition_count, last_text, text_was_filtered = _filter_segment(seg, text, repetition_count, last_text)
        if _should_strip_words(seg, text_was_filtered):
            seg.pop("words", None)

        processed_segments.append(seg)

    result["segments"] = processed_segments
    return result


def _extract_segments_or_none(result):
    if not result or "segments" not in result:
        return None
    segments = result["segments"]
    if not segments:
        return None
    return segments


def _filter_segment(seg, text, repetition_count, last_text):
    prob = seg.get("probability", 1.0)
    if _is_low_confidence(prob):
        _drop_segment_text(seg, "[Filter] Dropped segment due to low confidence (%.2f)", prob)
        return repetition_count, last_text, True

    if _contains_hallucination_phrase(text):
        _drop_segment_text(seg, "[Filter] Dropped segment containing hallucination phrase")
        return repetition_count, last_text, True

    repetition_count, last_text, repeated = _apply_repetition_filter(seg, text, repetition_count, last_text)
    return repetition_count, last_text, repeated


def _is_low_confidence(prob: float) -> bool:
    return prob < config.HALLUCINATION_SILENCE_THRESHOLD


def _contains_hallucination_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in config.HALLUCINATION_PHRASES)


def _apply_repetition_filter(seg, text, repetition_count, last_text):
    if text == last_text and text != "":
        repetition_count += 1
        if repetition_count >= config.HALLUCINATION_REPETITION_THRESHOLD:
            _drop_segment_text(seg, "[Filter] Dropped repetitive segment")
            return repetition_count, last_text, True
        return repetition_count, last_text, False

    return 0, text, False


def _drop_segment_text(seg, msg: str, *args):
    seg["text"] = ""
    logger.debug(msg, *args)


def _should_strip_words(seg, text_was_filtered: bool) -> bool:
    return text_was_filtered or not seg.get("text", "").strip()

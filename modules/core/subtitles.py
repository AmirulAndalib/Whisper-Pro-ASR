"""
Subtitle generation and text wrapping helper utilities.
"""

from modules.core import config


def wrap_text(text, max_line_width, max_line_count=None):
    """Wraps text to max_line_width characters per line, up to max_line_count lines."""
    if not text or not max_line_width:
        return text

    lines = []
    current_line = []
    current_len = 0
    for word in text.split():
        current_line, current_len = _append_wrapped_word(word, current_line, current_len, lines, max_line_width)
    _finalize_wrapped_line(current_line, lines)
    return "\n".join(_truncate_wrapped_lines(lines, max_line_count))


def _append_wrapped_word(word, current_line, current_len, lines, max_line_width):
    word_len = len(word)
    needed_space = word_len + (1 if current_line else 0)
    if current_len + needed_space <= max_line_width:
        current_line.append(word)
        return current_line, current_len + needed_space
    _finalize_wrapped_line(current_line, lines)
    return [word], word_len


def _finalize_wrapped_line(current_line, lines):
    if current_line:
        lines.append(" ".join(current_line))


def _truncate_wrapped_lines(lines, max_line_count):
    if not max_line_count:
        return lines
    return lines[:max_line_count]


def format_single_srt_block(idx, start_ts, end_ts, text, *, speaker=None, max_line_width=None, max_line_count=None):
    """Format a single subtitle segment into its SRT block representation."""
    start_fmt = format_timestamp(start_ts or 0.0)
    end_fmt = format_timestamp(end_ts or 0.0)
    clean_text = text.strip()
    if speaker:
        clean_text = f"[{speaker}]: {clean_text}"
    if max_line_width is not None:
        clean_text = wrap_text(clean_text, max_line_width, max_line_count)
    return f"{idx}\n{start_fmt} --> {end_fmt}\n{clean_text}\n\n"


def format_srt_highlighted_blocks(segment):
    """
    Generate sub-blocks for SRT highlighting where each word is shown
    in sequence with a highlight tag.
    """
    words = segment.get("words", [])
    if not words:
        return []

    segment_start_ts, segment_end_ts = _resolve_segment_timestamps(segment, default_end=0.0)
    speaker = segment.get("speaker")
    sub_blocks = []
    for active_idx, active_word in enumerate(words):
        start_ts, end_ts = _resolve_word_timestamps(active_word, segment_start_ts, segment_end_ts)
        text = _build_highlighted_word_line(words, active_idx)
        sub_blocks.append((start_ts, end_ts, _apply_speaker_prefix(text, speaker)))
    return sub_blocks


def _resolve_word_timestamps(active_word, segment_start_ts: float, segment_end_ts: float):
    start_ts = active_word.get("start", segment_start_ts)
    end_ts = active_word.get("end", segment_end_ts)
    return _coalesce_timestamp(start_ts, segment_start_ts), _coalesce_timestamp(end_ts, segment_end_ts)


def _build_highlighted_word_line(words, active_idx: int) -> str:
    text_parts = []
    for i, word_data in enumerate(words):
        text_parts.append(_render_highlighted_word(word_data.get("word", ""), i == active_idx))
    return "".join(text_parts).strip()


def _render_highlighted_word(word_text: str, is_active: bool) -> str:
    leading_spaces = len(word_text) - len(word_text.lstrip())
    space_prefix = word_text[:leading_spaces]
    clean_word = word_text[leading_spaces:]
    if is_active:
        return f'{space_prefix}<font color="#E0E0E0">{clean_word}</font>'
    return f"{space_prefix}{clean_word}"


def _get_normalized_segments(result):
    """Normalize the result segments and prepend the promo segment if enabled."""
    original_segments, used_fallback = _extract_or_build_fallback_segments(result)
    segments = [dict(seg) for seg in original_segments]
    if not _promo_is_enabled():
        return segments
    if used_fallback and segments:
        _shift_fallback_segment_after_promo(segments[0])
    promo_seg = {"start": 0.0, "end": config.SUBTITLE_PROMO_DURATION, "text": config.SUBTITLE_PROMO_TEXT}
    return [promo_seg] + segments


def _extract_or_build_fallback_segments(result) -> tuple[list, bool]:
    if _has_explicit_segments(result):
        return result.get("segments", []), False
    return _build_text_fallback_segments(result), True


def _has_explicit_segments(result) -> bool:
    return bool(result and isinstance(result, dict) and "segments" in result and result.get("segments"))


def _build_text_fallback_segments(result) -> list[dict]:
    text_val = result.get("text", "").strip() if (result and isinstance(result, dict)) else ""
    fallback_text = text_val if text_val else "[No dialogue detected]"
    return [{"start": 0.0, "end": 5.0, "text": fallback_text}]


def _promo_is_enabled() -> bool:
    return bool(config.SUBTITLE_PROMO_ENABLED and config.SUBTITLE_PROMO_TEXT)


def _shift_fallback_segment_after_promo(segment: dict):
    segment["start"] = config.SUBTITLE_PROMO_DURATION
    segment["end"] = config.SUBTITLE_PROMO_DURATION + 5.0


def generate_srt(result, max_line_width=None, max_line_count=None, highlight_words=False):
    """
    Compose industrial-standard SubRip (SRT) content from segment metadata.

    Handles time formatting, sequence indexing, and empty signal fallbacks.
    """
    segments = _get_normalized_segments(result)
    srt_lines, block_idx = [], 1
    for segment in segments:
        block_idx = _append_srt_segment_blocks(
            srt_lines,
            block_idx,
            segment,
            highlight_words,
            max_line_width=max_line_width,
            max_line_count=max_line_count,
        )

    if not srt_lines:
        return "1\n00:00:00,000 --> 00:00:05,000\n[No dialogue detected]\n"

    return "\n".join(srt_lines)


def _append_srt_segment_blocks(srt_lines: list, block_idx: int, segment: dict, highlight_words: bool, *, max_line_width, max_line_count):
    words = segment.get("words", [])
    if highlight_words and words:
        return _append_highlighted_srt_blocks(srt_lines, block_idx, segment, max_line_width, max_line_count)
    return _append_plain_srt_block(srt_lines, block_idx, segment, max_line_width, max_line_count)


def _append_highlighted_srt_blocks(srt_lines: list, block_idx: int, segment: dict, max_line_width, max_line_count) -> int:
    sub_blocks = format_srt_highlighted_blocks(segment)
    for start_ts, end_ts, text in sub_blocks:
        text = _maybe_wrap_text(text, max_line_width, max_line_count)
        srt_lines.append(f"{block_idx}\n{format_timestamp(start_ts)} --> {format_timestamp(end_ts)}\n{text}\n")
        block_idx += 1
    return block_idx


def _append_plain_srt_block(srt_lines: list, block_idx: int, segment: dict, max_line_width, max_line_count) -> int:
    start_ts, end_ts = _resolve_segment_timestamps(segment, default_end=5.0)
    text = _apply_speaker_prefix(segment.get("text", "").strip(), segment.get("speaker"))
    text = _maybe_wrap_text(text, max_line_width, max_line_count)
    srt_lines.append(f"{block_idx}\n{format_timestamp(start_ts)} --> {format_timestamp(end_ts)}\n{text}\n")
    return block_idx + 1


def format_timestamp(seconds):
    """Generate the millisecond-precision timestamp required for SRT specifications."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_vtt_karaoke(words, default_start_ts):
    """Format WebVTT karaoke timestamps for words list."""
    words_formatted = []
    for w in words:
        word_start_ts = w.get("start", default_start_ts)
        if word_start_ts is None:
            word_start_ts = default_start_ts
        w_start_fmt = format_timestamp(word_start_ts).replace(",", ".")
        word_text = w.get("word", "")
        leading_spaces = len(word_text) - len(word_text.lstrip())
        space_prefix = word_text[:leading_spaces]
        clean_word = word_text[leading_spaces:]
        words_formatted.append(f"{space_prefix}<{w_start_fmt}>{clean_word}")
    return "".join(words_formatted).strip()


def generate_vtt(result, max_line_width=None, max_line_count=None, highlight_words=False):
    """
    Generate WebVTT content for web-native subtitles.
    """
    segments = _get_normalized_segments(result)

    vtt_lines = ["WEBVTT", ""]
    for idx, segment in enumerate(segments, start=1):
        vtt_lines.append(_build_vtt_block(idx, segment, highlight_words, max_line_width, max_line_count))

    return "\n".join(vtt_lines)


def _build_vtt_block(idx: int, segment: dict, highlight_words: bool, max_line_width, max_line_count) -> str:
    start_ts, end_ts = _resolve_segment_timestamps(segment, default_end=5.0)
    start_fmt = format_timestamp(start_ts).replace(",", ".")
    end_fmt = format_timestamp(end_ts).replace(",", ".")
    text = _build_vtt_text(segment, highlight_words, start_ts)
    text = _apply_speaker_prefix(text, segment.get("speaker"))
    text = _maybe_wrap_text(text, max_line_width, max_line_count)
    return f"{idx}\n{start_fmt} --> {end_fmt}\n{text}\n"


def _build_vtt_text(segment: dict, highlight_words: bool, start_ts: float) -> str:
    words = segment.get("words", [])
    if highlight_words and words:
        return _format_vtt_karaoke(words, start_ts)
    return segment.get("text", "").strip()


def generate_txt(result):
    """
    Generate plain text transcript.
    """
    if not result:
        return ""
    segments = result.get("segments")
    if segments:
        txt_lines = []
        for segment in segments:
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker")
            if speaker:
                txt_lines.append(f"[{speaker}]: {text}")
            else:
                txt_lines.append(text)
        return "\n".join(txt_lines)
    return result.get("text", "").strip()


def generate_tsv(result):
    """
    Generate Tab-Separated Values (TSV) format.
    """
    if not result:
        return "start\tend\ttext"

    tsv_lines = ["start\tend\ttext"]
    for segment in result.get("segments", []):
        try:
            start_ts, end_ts = _resolve_segment_timestamps(segment, default_end=0.0)
            start_ms = int(start_ts * 1000)
            end_ms = int(end_ts * 1000)
            text = _sanitize_tsv_text(_apply_speaker_prefix(segment.get("text", "").strip(), segment.get("speaker")))
            tsv_lines.append(f"{start_ms}\t{end_ms}\t{text}")
        except (TypeError, ValueError, KeyError):
            continue

    return "\n".join(tsv_lines)


def _resolve_segment_timestamps(segment: dict, default_end: float) -> tuple[float, float]:
    try:
        if "timestamp" in segment:
            start_ts, end_ts = segment["timestamp"]
        else:
            start_ts = segment.get("start", 0.0)
            end_ts = segment.get("end", default_end)
        return _coalesce_timestamp(start_ts, 0.0), _coalesce_timestamp(end_ts, default_end)
    except (ValueError, TypeError, KeyError):
        return 0.0, default_end


def _coalesce_timestamp(value, fallback: float) -> float:
    if value is None:
        return fallback
    return value


def _apply_speaker_prefix(text: str, speaker) -> str:
    if speaker:
        return f"[{speaker}]: {text}"
    return text


def _maybe_wrap_text(text: str, max_line_width, max_line_count) -> str:
    if max_line_width is not None:
        return wrap_text(text, max_line_width, max_line_count)
    return text


def _sanitize_tsv_text(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ")

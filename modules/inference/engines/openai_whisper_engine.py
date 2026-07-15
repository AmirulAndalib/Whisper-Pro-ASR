"""OpenAI Whisper engine wrapper."""

import importlib
from typing import Any, Optional

from modules.inference.engines.base import BaseASREngine, build_inference_info, iter_segment_wrappers


class OpenaiWhisperEngine(BaseASREngine):
    """Standard PyTorch openai-whisper engine."""

    def __init__(self, model_id: str, device: str):
        self.whisper = importlib.import_module("whisper")
        self.device = device
        self.model = self.whisper.load_model(model_id, device=device)

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
            "initial_prompt": initial_prompt,
            "word_timestamps": word_timestamps,
        }
        for key in [
            "beam_size",
            "best_of",
            "patience",
            "length_penalty",
            "temperature",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "fp16",
        ]:
            if key in kwargs:
                params[key] = kwargs[key]

        result = self.model.transcribe(audio_path, language=language, task=task, **params)
        return iter_segment_wrappers(result), build_inference_info(result, audio_path, language)

    def detect_language(self, audio: Any):
        """Identify language using OpenAI Whisper language head."""
        if isinstance(audio, str):
            audio = self.whisper.load_audio(audio)

        mel = self.whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        lang_code, lang_prob = ordered[0]
        return lang_code, float(lang_prob), [(k, float(v)) for k, v in ordered]

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model

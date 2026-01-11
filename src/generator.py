""" Answer generation module using Hugging Face transformers. """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import pipeline

from src.prompts import REFUSAL


@dataclass
class GenerationResult:
    """ Result of a text generation call. """
    text: str
    raw: Any


class AnswerGenerator:
    """
    Thin wrapper around a HF generation pipeline.
    Designed for text2text models like FLAN-T5.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: int = -1,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        generate_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.generate_kwargs = generate_kwargs or {}

        self._pipe = pipeline(
            task="text2text-generation",
            model=self.model_name,
            device=self.device,
        )

    def generate(self, prompt: str, generate_kwargs: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """ Generate an answer given the input prompt. """
        # base kwargs from init + per-call overrides
        kwargs = dict(self.generate_kwargs)
        if generate_kwargs:
            kwargs.update(generate_kwargs)

        # Core generation controls
        kwargs.setdefault("max_new_tokens", self.max_new_tokens)
        kwargs.setdefault("do_sample", self.do_sample)
        kwargs.setdefault("temperature", self.temperature)

        out = self._pipe(prompt, **kwargs)

        text = ""
        if isinstance(out, list) and out:
            text = (out[0].get("generated_text") or "").strip()
        else:
            text = str(out).strip()

        return GenerationResult(text=self._postprocess(text), raw=out)

    def _postprocess(self, text: str) -> str:
        if not text or not text.strip():
            return REFUSAL

        # Preserve newlines (important for bullet lists); normalize whitespace per line
        lines = [" ".join(ln.split()).strip() for ln in text.splitlines()]
        cleaned = "\n".join([ln for ln in lines if ln]).strip()

        # Snap exact refusal if it matches case-insensitively (keeps checks consistent)
        if cleaned.lower().strip('"').strip("'") == REFUSAL.lower():
            return REFUSAL

        return cleaned

"""
Google model adapter for Phase 3 evaluation.
Handles: gemini-2.5-flash-lite (B1,B3), gemini-3.1-pro (B0,B1,B3)
Uses Google GenAI SDK via Vertex AI.
"""
from __future__ import annotations

import base64
import time
import threading
import logging
from typing import Optional

from .base_adapter import BaseModelAdapter

log = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = (
    "Answer the question about the given document image. "
    "Give a short, precise answer. "
    "If the answer is a number, return just the number. "
    "If it is a name or short text, return just that text. "
    "Be concise."
)

ANSWER_HEADROOM = 512   # tokens reserved for answer after thinking
MAX_RETRIES = 3
BASE_BACKOFF = 2.0

BUDGET_TO_THINKING = {
    "B0": None,       # No thinking (Gemini Pro only)
    "B1": 1024,
    "B3": 16384,
}

_google_client = None
_client_lock = threading.Lock()


def _get_google_client():
    global _google_client
    if _google_client is None:
        with _client_lock:
            if _google_client is None:
                import os
                from google import genai
                _google_client = genai.Client(
                    vertexai=True,
                    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "speedy-aurora-193605"),
                    location=os.environ.get("VERTEX_AI_REGION", "us-central1"),
                )
    return _google_client


class GoogleAdapter(BaseModelAdapter):
    """Adapter for Gemini 2.5 Flash Lite and Gemini 3.1 Pro."""

    def call(self, image_b64: str, query: str) -> dict:
        model_id = self.config["model_id"]
        thinking_budget = BUDGET_TO_THINKING.get(self.budget_level)

        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call_once(model_id, image_b64, query, thinking_budget)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                is_rate = any(kw in exc_str.lower() for kw in ["429", "rate", "quota", "resource_exhausted"])
                backoff = BASE_BACKOFF * (2 ** (attempt - 1))
                log.warning(
                    "%s attempt %d/%d %s. Sleeping %.1fs.",
                    self.config_id, attempt, MAX_RETRIES,
                    "rate-limited" if is_rate else f"error: {exc_str[:80]}",
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(f"All retries failed for {self.config_id}") from last_exc

    def _call_once(self, model_id: str, image_b64: str, query: str, thinking_budget: Optional[int]) -> dict:
        from google.genai import types as genai_types

        client = _get_google_client()

        img_bytes = base64.b64decode(image_b64)
        contents = [
            genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            genai_types.Part.from_text(text=query),
        ]

        gen_config_kwargs = dict(
            system_instruction=EVAL_SYSTEM_PROMPT,
            max_output_tokens=self.budget_tokens + ANSWER_HEADROOM,
            temperature=0.0,
        )

        # Add thinking config for B1/B3
        if thinking_budget is not None:
            gen_config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget,
            )

        gen_config = genai_types.GenerateContentConfig(**gen_config_kwargs)

        t0 = time.monotonic()
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=gen_config,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Extract answer (non-thought parts only)
        answer = ""
        for part in response.candidates[0].content.parts:
            if not (hasattr(part, "thought") and part.thought):
                answer += part.text or ""
        answer = answer.strip()

        # Extract token counts
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        reasoning_tokens = getattr(usage, "thoughts_token_count", 0) or 0

        return {
            "answer": answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }

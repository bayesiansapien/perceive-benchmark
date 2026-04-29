"""
OpenAI model adapter for Phase 3 evaluation.
Handles: gpt-5.4-nano (B0), gpt-5.4-mini (B0-B3), gpt-5.4 (B0-B3)
"""
from __future__ import annotations

import os
import time
import threading
import logging
from pathlib import Path
from typing import Optional

from .base_adapter import BaseModelAdapter

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

log = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = (
    "Answer the question about the given document image. "
    "Give a short, precise answer. "
    "If the answer is a number, return just the number. "
    "If it is a name or short text, return just that text. "
    "Be concise."
)

ANSWER_HEADROOM = 512   # tokens reserved for the answer after thinking
MAX_RETRIES = 3
BASE_BACKOFF = 2.0

BUDGET_TO_EFFORT = {
    "B0": None,
    "B1": "low",
    "B2": "medium",
    "B3": "high",
}

_openai_client = None
_client_lock = threading.Lock()


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                import openai
                _openai_client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
    return _openai_client


class OpenAIAdapter(BaseModelAdapter):
    """Adapter for GPT-5.4-nano, GPT-5.4-mini, GPT-5.4."""

    def call(self, image_b64: str, query: str) -> dict:
        """Call OpenAI API with retry. Returns raw inference dict."""
        model_id = self.config["model_id"]
        effort = BUDGET_TO_EFFORT.get(self.budget_level)

        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call_once(model_id, image_b64, query, effort)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                is_rate = any(kw in exc_str.lower() for kw in ["429", "rate", "quota"])
                backoff = BASE_BACKOFF * (2 ** (attempt - 1))
                log.warning(
                    "%s attempt %d/%d %s. Sleeping %.1fs.",
                    self.config_id, attempt, MAX_RETRIES,
                    "rate-limited" if is_rate else f"error: {exc_str[:80]}",
                    backoff,
                )
                time.sleep(backoff)

        log.error("Giving up on %s after %d attempts: %s", self.config_id, MAX_RETRIES, str(last_exc))
        raise RuntimeError(f"All retries failed for {self.config_id}") from last_exc

    def _call_once(self, model_id: str, image_b64: str, query: str, effort: Optional[str]) -> dict:
        client = _get_openai_client()

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ]

        # max_completion_tokens must cover reasoning tokens + answer tokens.
        # At B3 (16384 reasoning budget), the model can use up to 16384 tokens
        # for thinking before generating the answer. A fixed 512 cap causes
        # empty answers when thinking consumes the full budget.
        max_tokens = self.budget_tokens + ANSWER_HEADROOM

        kwargs = dict(
            model=model_id,
            messages=messages,
            max_completion_tokens=max_tokens,
        )
        if effort is not None:
            kwargs["reasoning_effort"] = effort

        t0 = time.monotonic()
        response = client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        answer = (response.choices[0].message.content or "").strip()
        usage = response.usage

        # Extract reasoning tokens (available when thinking is enabled)
        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

        return {
            "answer": answer,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }

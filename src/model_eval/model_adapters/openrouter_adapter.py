"""
OpenRouter adapter — Qwen3-VL-Plus (B0-B3) and Llama 4 Scout (B0 only).

Both models are accessed via OpenRouter's OpenAI-compatible gateway.
Required env var: OPENROUTER_API_KEY

Qwen3-VL-Plus thinking budget is passed via extra_body (DashScope passthrough):
  B0: enable_thinking=False
  B1/B2/B3: enable_thinking=True, thinking_budget=<tokens>

Llama 4 Scout has no thinking support — treated as B0-only like gpt-5.4-nano.
"""
from __future__ import annotations

import os
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

ANSWER_HEADROOM = 512
MAX_RETRIES = 5
BASE_BACKOFF = 3.0

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import openai
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENROUTER_API_KEY not set")
                _client = openai.OpenAI(
                    api_key=api_key,
                    base_url=OPENROUTER_BASE_URL,
                )
    return _client


class OpenRouterAdapter(BaseModelAdapter):
    """Adapter for models served via OpenRouter (Qwen3-VL-Plus, Llama 4 Scout)."""

    def call(self, image_b64: str, query: str) -> dict:
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._call_once(image_b64, query)
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                is_rate = any(kw in exc_str.lower() for kw in ["429", "rate", "quota", "too many"])
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

    def _call_once(self, image_b64: str, query: str) -> dict:
        client = _get_client()
        model_id = self.config["model_id"]
        model_family = self.config.get("model_family", "")

        messages = [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]

        max_tokens = self.budget_tokens + ANSWER_HEADROOM

        kwargs: dict = dict(
            model=model_id,
            messages=messages,
            max_completion_tokens=max_tokens,
        )

        # Qwen3-VL: B0 uses instruct model, B1-B3 use thinking model + thinking_budget
        if model_family == "qwen":
            if self.budget_tokens == 0:
                pass  # model_id already set to instruct variant
            else:
                thinking_model_id = self.config.get("model_id_thinking", model_id)
                kwargs["model"] = thinking_model_id
                kwargs["extra_body"] = {"thinking_budget": self.budget_tokens}

        # Llama 4 Scout: no extra params needed (B0 only, standard call)

        t0 = time.monotonic()
        response = client.chat.completions.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        msg = response.choices[0].message

        # Qwen returns thinking in reasoning_content, final answer in content
        answer = (msg.content or "").strip()
        reasoning_content = getattr(msg, "reasoning_content", None) or ""

        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(usage, "completion_tokens", 0) or 0

        # Reasoning tokens: from completion_tokens_details if available
        reasoning_tokens = 0
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        # Fallback: estimate from reasoning_content length
        if reasoning_tokens == 0 and reasoning_content:
            reasoning_tokens = len(reasoning_content.split())

        return {
            "answer": answer,
            "reasoning_content": reasoning_content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }

"""
Anthropic model adapter for Phase 3 evaluation.
Handles: claude-sonnet-4-6 (B0-B3), claude-opus-4-6 (B0,B1,B3)
Uses Anthropic SDK via Vertex AI.
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

MAX_OUTPUT_TOKENS = 16384 + 512  # must be > thinking budget + answer tokens
MAX_RETRIES = 3
BASE_BACKOFF = 2.0

BUDGET_TO_THINKING = {
    "B0": None,
    "B1": 1024,
    "B2": 4096,
    "B3": 16384,
}

_anthropic_client = None
_client_lock = threading.Lock()


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        with _client_lock:
            if _anthropic_client is None:
                # Use AnthropicVertex when ANTHROPIC_VERTEX_PROJECT_ID is set (Vertex AI env)
                # Region from CLOUD_ML_REGION env var (defaults to "global" for Anthropic Vertex)
                vertex_project = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
                vertex_region = os.environ.get("CLOUD_ML_REGION", "global")
                if vertex_project:
                    from anthropic import AnthropicVertex
                    _anthropic_client = AnthropicVertex(
                        project_id=vertex_project,
                        region=vertex_region,
                    )
                else:
                    import anthropic
                    _anthropic_client = anthropic.Anthropic(
                        api_key=os.environ.get("ANTHROPIC_API_KEY"),
                    )
    return _anthropic_client


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Claude Sonnet 4.6 and Claude Opus 4.6."""

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
                is_rate = any(kw in exc_str.lower() for kw in ["429", "rate", "quota", "overloaded"])
                backoff = BASE_BACKOFF * (2 ** (attempt - 1))
                log.warning(
                    "%s attempt %d/%d %s. Sleeping %.1fs.",
                    self.config_id, attempt, MAX_RETRIES,
                    "rate-limited" if is_rate else f"error: {exc_str[:80]}",
                    backoff,
                )
                time.sleep(backoff)

        raise RuntimeError(f"All retries failed for {self.config_id}") from last_exc

    def _call_once(
        self,
        model_id: str,
        image_b64: str,
        query: str,
        thinking_budget: Optional[int],
    ) -> dict:
        client = _get_anthropic_client()

        # Build content blocks: image first, then the question
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": query},
        ]

        kwargs = dict(
            model=model_id,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=EVAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        # Enable extended thinking for B1/B2/B3
        # Note: no 'betas' kwarg, not supported on AnthropicVertex
        if thinking_budget is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        t0 = time.monotonic()
        response = client.messages.create(**kwargs)
        latency_ms = int((time.monotonic() - t0) * 1000)

        # Separate thinking blocks from answer blocks
        thinking_text = ""
        answer_text = ""

        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "thinking":
                thinking_text += getattr(block, "thinking", "")
            elif block_type == "text":
                answer_text += getattr(block, "text", "")

        # Estimate reasoning tokens from thinking content length
        # Anthropic doesn't separately report thinking tokens in all SDK versions;
        # chars / 4 is a standard approximation.
        reasoning_tokens = len(thinking_text) // 4 if thinking_text else 0

        usage = response.usage
        return {
            "answer": answer_text.strip(),
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "reasoning_tokens": reasoning_tokens,
            "latency_ms": latency_ms,
        }

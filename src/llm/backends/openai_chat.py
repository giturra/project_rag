from __future__ import annotations

import time
from typing import Optional, Sequence

from openai import OpenAI

from ..types import Message, LLMResult, Usage


class OpenAIChatLLM:
    """
    OpenAI Chat backend implementing BaseLLM interface.

    Compatible with:
      - gpt-3.5-turbo-16k
      - gpt-4.x
      - future chat-completions-compatible models
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout_s: float = 60.0,
        organization: Optional[str] = None,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.model = model
        self.backend = "openai"

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout_s,
        )

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> LLMResult:
        t0 = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

        latency_ms = (time.time() - t0) * 1000.0

        # --- Extract text ---
        text = response.choices[0].message.content or ""

        # --- Normalize usage (robust to None / missing fields) ---
        u = response.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )

        return LLMResult(
            text=text,
            usage=usage,
            raw=response,            # keep full object for debugging
            backend=self.backend,
            model=self.model,
            latency_ms=latency_ms,
        )
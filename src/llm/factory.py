from __future__ import annotations

import os
from typing import Optional

from .config import LLMConfig
from .base import BaseLLM


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def build_llm(cfg: LLMConfig) -> BaseLLM:
    """
    Create and return an LLM backend instance based on cfg.

    Rules:
      - CLI should have already filled cfg with values.
      - Secrets (API keys) may still come from env as fallback.
      - Keep backend-specific imports inside branches.
    """
    backend = cfg.backend.lower()

    if backend == "openai":
        from .backends.openai_chat import OpenAIChatLLM

        api_key = cfg.api_key or _env("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. "
                "Set OPENAI_API_KEY or pass --api-key."
            )

        base_url = cfg.base_url or _env("OPENAI_BASE_URL")  # optional

        return OpenAIChatLLM(
            model=cfg.model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=cfg.timeout_s,
        )

    # if backend == "dummy":
    #     from .backends.dummy import DummyLLM
    #     return DummyLLM(model=cfg.model)

    raise ValueError(f"Unknown backend: {cfg.backend}")

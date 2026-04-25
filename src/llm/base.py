from __future__ import annotations
from typing import Optional, Protocol, Sequence
from .types import Message, LLMResult

class BaseLLM(Protocol):
    
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
    ) -> LLMResult:
        ...

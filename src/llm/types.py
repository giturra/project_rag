from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]

@dataclass(frozen=True)
class Message:
    role: Role
    content: str

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMResult:
    text: str
    usage: Usage = Usage()
    raw: Any = None
    backend: str = ""
    model: str = ""
    latency_ms: float = 0.0

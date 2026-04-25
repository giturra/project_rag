from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

Backend = Literal["openai", "ollama"]

@dataclass
class LLMConfig:
    backend: Backend
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: float = 60.0
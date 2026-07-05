# domain/llm.py
from __future__ import annotations
from typing import List, Protocol, TypedDict

class Msg(TypedDict):
    role: str   # "system" | "user" | "assistant" | "tool"
    content: str

class LLMResult(TypedDict, total=False):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

class ILLMProvider(Protocol):
    def generate(self, messages: List[Msg], temperature: float = 0.3) -> LLMResult: ...

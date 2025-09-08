# app/prompting.py
from __future__ import annotations
from typing import List
from domain.llm import Msg

SYSTEM_DEFAULT = (
    "You are a precise assistant. Use given context. If conflicts are present, explain them briefly."
)

def make_messages(user_q: str, context_sections: List[str], system: str | None = None) -> List[Msg]:
    sys = system or SYSTEM_DEFAULT
    ctx_text = "\n\n".join(context_sections)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Question: {user_q}\n\nContext:\n{ctx_text}\n\nAnswer concisely."},
    ]

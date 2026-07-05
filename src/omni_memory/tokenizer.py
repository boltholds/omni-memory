# app/tokenizer.py
from __future__ import annotations
from typing import List, Protocol

class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...
    def count_many(self, texts: List[str]) -> List[int]: ...

# ---------- Фолбэк (быстро и просто) ----------
class SimpleTokenizer:
    """Приближённо: токены ~= слова + пунктуация."""
    def count(self, text: str) -> int:
        # грубо: split по whitespace; пустое => 0
        return 0 if not text else len(text.split())

    def count_many(self, texts: List[str]) -> List[int]:
        return [self.count(t) for t in texts]

# ---------- Tiktoken (если доступен) ----------
class TiktokenTokenizer:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        try:
            import tiktoken  # type: ignore
        except Exception as e:
            raise RuntimeError("tiktoken недоступен; используйте SimpleTokenizer") from e
        try:
            self._enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            # универсальная кодировка для многих моделей
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text or ""))

    def count_many(self, texts: List[str]) -> List[int]:
        return [len(self._enc.encode(t or "")) for t in texts]

# ---------- Фабрика ----------
def build_tokenizer(backend: str = "auto", model_name: str = "gpt-4o-mini") -> Tokenizer:
    if backend == "simple":
        return SimpleTokenizer()
    if backend == "tiktoken":
        return TiktokenTokenizer(model_name)
    # auto
    try:
        return TiktokenTokenizer(model_name)
    except Exception:
        return SimpleTokenizer()

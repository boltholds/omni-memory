from typing import List

class WorkingMemory:
    def __init__(self, max_tokens: int = 200):
        self.max_tokens = max_tokens
        self.buffer: List[str] = []

    def append(self, text: str) -> None:
        """Добавляем новый кусок в память"""
        self.buffer.append(text)
        self.truncate()

    def truncate(self) -> None:
        """Усечение до лимита токенов (пока считаем токен = слово)"""
        all_words = " ".join(self.buffer).split()
        if len(all_words) <= self.max_tokens:
            return
        # оставляем последние max_tokens слов
        trimmed = all_words[-self.max_tokens :]
        self.buffer = [" ".join(trimmed)]

    def to_prompt(self) -> str:
        """Собираем буфер в строку для LLM"""
        return "\n".join(self.buffer)

from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any
from domain.llm import Msg
from app.config import settings
import os

Lang = Literal["en","ru"]
Style = Literal["concise","bullets","detailed"]

# Фолбэк (как раньше), если нет Jinja или шаблонов
SYSTEM_FALLBACK = """You are a precise assistant. Use ONLY the provided context.
- If facts conflict, mention that briefly.
- Be helpful, neutral, and avoid speculation.
- If the answer is unknown from context, say so."""
STYLE_HINTS = {"concise":"Answer in 1–2 sentences.","bullets":"Answer as 3–5 bullet points.","detailed":"Answer in a short, structured paragraph."}
LANG_HINTS = {"en":"Respond in English.","ru":"Отвечай по-русски."}

class PromptRenderer:
    def __init__(
        self,
        template_dir: Optional[str] = None,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        self.template_dir = (template_dir or settings.prompt_template_dir).rstrip("/\\")
        self.system_template = system_template or settings.prompt_system_template
        self.user_template = user_template or settings.prompt_user_template

        self._jinja_env = None
        try:
            from jinja2 import Environment, FileSystemLoader, StrictUndefined
            if os.path.isdir(self.template_dir):
                self._jinja_env = Environment(
                    loader=FileSystemLoader(self.template_dir),
                    autoescape=False,
                    trim_blocks=True,
                    lstrip_blocks=True,
                    undefined=StrictUndefined,  # пусть падает, если переменная не определена
                )
        except Exception:
            self._jinja_env = None  # фолбэк

    def make_messages(
        self,
        user_q: str,
        context_sections: List[str],
        lang: Lang = "en",
        style: Style = "concise",
        extra: Optional[Dict[str, Any]] = None,
    ) -> List[Msg]:
        """
        Рендерит два сообщения: system + user.
        Переменные, доступные в шаблонах:
          - q: строка вопроса пользователя
          - context_sections: List[str] (каждый элемент — уже отформатированная секция "Title:\nBody")
          - lang: "en"|"ru"
          - style: "concise"|"bullets"|"detailed"
          - style_hint / lang_hint: короткие подсказки
          - extra: произвольный dict для расширений
        """
        extra = extra or {}
        ctx_text = "\n\n".join(context_sections) if context_sections else "(no context)"
        style_hint = STYLE_HINTS.get(style, STYLE_HINTS["concise"])
        lang_hint = LANG_HINTS.get(lang, LANG_HINTS["en"])

        values = {
            "q": user_q,
            "context_sections": context_sections,
            "context_text": ctx_text,
            "lang": lang,
            "style": style,
            "style_hint": style_hint,
            "lang_hint": lang_hint,
            "extra": extra,
        }

        if self._jinja_env:
            try:
                sys_t = self._jinja_env.get_template(self.system_template)
                usr_t = self._jinja_env.get_template(self.user_template)
                system_text = sys_t.render(**values).strip()
                user_text = usr_t.render(**values).strip()
                return [{"role":"system","content": system_text},
                        {"role":"user","content": user_text}]
            except Exception:
                pass  # на любой сбой — фолбэк

        # ---- ФОЛБЭК ----
        system_text = SYSTEM_FALLBACK
        user_text = (
            f"Question:\n{user_q}\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Instructions: {style_hint} {lang_hint}"
        )
        return [{"role":"system","content": system_text},
                {"role":"user","content": user_text}]

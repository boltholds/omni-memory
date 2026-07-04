from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any
from domain.llm import Msg
from app.config import settings
import logging
import os

Lang = Literal["en","ru"]
Style = Literal["concise","bullets","detailed"]
log = logging.getLogger("app.prompting")

# Фолбэк (как раньше), если нет Jinja или шаблонов
SYSTEM_FALLBACK = """You are a precise assistant. Use ONLY the provided context.
- If facts conflict, mention that briefly.
- Be helpful, neutral, and avoid speculation.
- If the answer is unknown from context, say so."""
MEMORY_GROUNDED_CONTRACT = """Memory-grounded answer contract:
- Before choosing an action, identify the most relevant memory section.
- If memory describes a previous failure pattern, forbidden approach, accepted decision, or reusable skill, your plan must follow it.
- If the user's suggested shortcut conflicts with relevant memory, reject the shortcut and explain the safer action.
- Do not recommend an action that repeats a remembered mistake."""
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
        except Exception as exc:
            log.warning(
                "prompt_template_environment_unavailable",
                exc_info=True,
                extra={
                    "component": "PromptRenderer",
                    "op": "init_jinja_environment",
                    "template_dir": self.template_dir,
                    "error_type": type(exc).__name__,
                    "fallback": "builtin_prompt",
                },
            )
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
            except Exception as exc:
                log.warning(
                    "prompt_template_render_failed",
                    exc_info=True,
                    extra={
                        "component": "PromptRenderer",
                        "op": "render_jinja_prompt",
                        "template_dir": self.template_dir,
                        "system_template": self.system_template,
                        "user_template": self.user_template,
                        "error_type": type(exc).__name__,
                        "fallback": "builtin_prompt",
                    },
                )

        # ---- ФОЛБЭК ----
        system_text = _system_text(context_sections, extra=extra)
        user_text = (
            f"Question:\n{user_q}\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"Instructions: {style_hint} {lang_hint}"
        )
        return [{"role":"system","content": system_text},
                {"role":"user","content": user_text}]


def _system_text(context_sections: List[str], *, extra: Dict[str, Any]) -> str:
    if extra.get("disable_memory_grounding_contract"):
        return SYSTEM_FALLBACK
    if not context_sections:
        return SYSTEM_FALLBACK
    return f"{SYSTEM_FALLBACK}\n\n{MEMORY_GROUNDED_CONTRACT}"

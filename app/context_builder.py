# app/context_builder.py
from __future__ import annotations
from typing import List, Tuple
from domain.models import RetrievalBundle, ContextPack, ContextSection

def _tok_count(s: str) -> int:
    # грубая оценка количества «токенов»: слова + пунктуация
    # этого достаточно для бюджетного усечения
    return max(0, len(s.split()))

def _take_lines_up_to_budget(lines: List[str], budget: int) -> Tuple[List[str], bool]:
    """
    Возвращает часть списка строк, которые укладываются в бюджет (по словам).
    Второе значение: был ли обрезан список (True = обрезали).
    """
    out, used = [], 0
    for line in lines:
        c = _tok_count(line)
        if used + c <= budget:
            out.append(line)
            used += c
        else:
            # попробуем частично урезать последнюю строку
            remain = max(0, budget - used)
            if remain > 0:
                parts = line.split()
                out.append(" ".join(parts[:remain]) + " …")
                used = budget
            return out, True
    return out, False

def _section(title: str, items: List[str]) -> ContextSection:
    return ContextSection(title=title, body="\n".join(f"- {x}" for x in items))

def build_context(bundle: RetrievalBundle, max_tokens: int) -> Tuple[ContextPack, List[str]]:
    """
    Собираем секции в порядке приоритета с учётом бюджета.
    Возвращает ContextPack и список advisories.
    """
    advisories: List[str] = []
    budget = max_tokens

    # Подготовим кандидатные строки по секциям
    conflicts_lines: List[str] = []
    # conflicts вычисляет Orchestrator; здесь — если bundle.citations содержит подсказки,
    # пропустим. Для совместимости оставляем пусто — наполнит Orchestrator.
    # Этот билдер отвечает только за бюджет/усечение, а не за детект конфликтов.

    facts_lines = [f"{f.subject} {f.predicate} {f.object}" for f in bundle.facts]
    episodes_lines = [e.summary.strip() or "(no summary)" for e in bundle.episodes]
    notes_lines = [
        (o.payload.get("text") or str(o.payload)).strip() for o in bundle.semantic_chunks
        if (o.payload or {})
    ]

    # Заведём результат
    sections: List[ContextSection] = []

    # 1) Conflicts — не заполняем здесь (см. Orchestrator), но зарезервируем слот: билдер сможет их усечь
    # вернём пустой список для обратной совместимости; Orchestrator подставит готовые строки.

    # 2) Facts
    if facts_lines:
        take, trimmed = _take_lines_up_to_budget(facts_lines, budget)
        sections.append(_section("Facts", take))
        used = _tok_count("\n".join(take))
        budget = max(0, budget - used)
        if trimmed:
            advisories.append("Facts trimmed to fit context budget.")

    # 3) Episodes
    if budget > 0 and episodes_lines:
        take, trimmed = _take_lines_up_to_budget(episodes_lines, budget)
        sections.append(_section("Episodes", take))
        used = _tok_count("\n".join(take))
        budget = max(0, budget - used)
        if trimmed:
            advisories.append("Episodes trimmed to fit context budget.")

    # 4) Semantic Notes (наименее приоритетные)
    if budget > 0 and notes_lines:
        take, trimmed = _take_lines_up_to_budget(notes_lines, budget)
        sections.append(_section("Semantic Notes", take))
        used = _tok_count("\n".join(take))
        budget = max(0, budget - used)
        if trimmed:
            advisories.append("Semantic Notes trimmed to fit context budget.")

    # Если бюджета изначально не хватило ни на что
    if not sections:
        advisories.append("Context budget too small; nothing included.")

    return ContextPack(sections=sections, advisories=advisories), advisories

# app/orchestrator.py
from __future__ import annotations
from typing import List

from domain.models import RetrievalBundle, ContextPack, ContextSection, ConflictReport
from domain.ports import IMemoryOrchestrator, IRetriever, IConsistencyEngine
from infra.metrics import metrics
from app.context_builder import build_context
from app.config import settings


class Orchestrator(IMemoryOrchestrator):
    def __init__(self, retriever: IRetriever, consistency: IConsistencyEngine):
        self._retriever = retriever
        self._consistency = consistency

    def plan_retrieval(self, query: str) -> RetrievalBundle:
        metrics.inc("context_calls", 1)
        return self._retriever.retrieve(query)

    def assemble_context(self, bundle: RetrievalBundle) -> ContextPack:
            # 1) Сначала строим базовые секции без конфликтов (по бюджету)
            pack, advisories = build_context(bundle, settings.context_max_tokens)

            # 2) Вставим секцию Conflicts с самым высоким приоритетом и пересоберём бюджет
            report = self._consistency.detect_conflicts(bundle.facts)
            if report.conflicts:
                # соберём строки конфликтов
                conflict_lines = [f"{c.key}: {', '.join(c.variants)}" for c in report.conflicts]
                # Вставим в начало списка секций
                conflicts_sec = ContextSection(
                    title="Conflicts",
                    body="\n".join(f"- {ln}" for ln in conflict_lines)
                )
                # Новый порядок: Conflicts → остальные из pack.sections (уже усечены)
                new_sections = [conflicts_sec] + pack.sections
                # Проверим бюджет заново: если конфликты «съели» всё — оставим хотя бы их
                # (простейшая стратегия: если перебор — обрежем хвостовые секции)
                total_text = "\n".join(sec.body for sec in new_sections)
                if len(total_text.split()) > settings.context_max_tokens:
                    # Обрежем до бюджета грубо, сохранив Conflicts полностью
                    remain = settings.context_max_tokens - len(conflicts_sec.body.split())
                    remain = max(0, remain)
                    trimmed_sections = [conflicts_sec]
                    for sec in pack.sections:
                        if remain <= 0:
                            break
                        # возьмём строки секции построчно, чтобы не переполнить
                        lines = [ln.lstrip("- ").strip() for ln in sec.body.splitlines()]
                        from app.context_builder import _take_lines_up_to_budget, _section
                        take, trimmed = _take_lines_up_to_budget(lines, remain)
                        if take:
                            trimmed_sections.append(_section(sec.title, take))
                            remain -= len(" ".join(take).split())
                            if trimmed:
                                advisories.append(f"{sec.title} trimmed after adding conflicts.")
                    pack.sections = trimmed_sections
                else:
                    pack.sections = new_sections

                advisories.append(f"Detected {len(report.conflicts)} conflict(s).")

            pack.advisories = list(dict.fromkeys(advisories))  # de-dup, preserve order
            return pack

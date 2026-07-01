# app/context_builder.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Protocol, Tuple

from app.config import settings
from app.tokenizer import build_tokenizer
from domain.models import ContextSection,ContextPack,RetrievalBundle
from domain.ports import IConsistencyEngine

_tok = build_tokenizer(settings.tokenizer_backend, settings.tokenizer_model)

def _tok_count(s: str) -> int:
    return _tok.count(s or "")

def _take_lines_up_to_budget(lines: List[str], budget: int) -> Tuple[List[str], bool]:
    out, used = [], 0
    for line in lines:
        c = _tok_count(line)
        if used + c <= budget:
            out.append(line)
            used += c
        else:
            # частично усечём строку приблизительно по словам — дешёво и эффективно
            parts = (line or "").split()
            # бинарный поиск по количеству слов, чтобы уложиться по токенам
            lo, hi, best = 0, len(parts), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = " ".join(parts[:mid])
                if used + _tok_count(candidate) <= budget:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best > 0:
                out.append(" ".join(parts[:best]) + " …")
            return out, True
    return out, False

def _section(title: str, items: List[str]) -> ContextSection:
    return ContextSection(title=title, body="\n".join(f"- {x}" for x in items))


@dataclass
class ContextBuildState:
    bundle: RetrievalBundle
    budget: int
    consistency: IConsistencyEngine | None = None
    sections: List[ContextSection] = field(default_factory=list)
    advisories: List[str] = field(default_factory=list)


class ContextBuildStrategy(Protocol):
    def handle(self, state: ContextBuildState) -> ContextBuildState:
        ...


class BudgetedSectionStrategy:
    title: str

    def lines(self, bundle: RetrievalBundle) -> List[str]:
        raise NotImplementedError

    def handle(self, state: ContextBuildState) -> ContextBuildState:
        lines = self.lines(state.bundle)
        if not lines:
            return state

        take, trimmed = _take_lines_up_to_budget(lines, state.budget)
        state.sections.append(_section(self.title, take))
        used = _tok_count("\n".join(take))
        state.budget = max(0, state.budget - used)

        if trimmed:
            state.advisories.append(f"{self.title} trimmed to fit context budget.")

        return state


class ConflictsStrategy(BudgetedSectionStrategy):
    title = "Conflicts"

    def lines(self, bundle: RetrievalBundle) -> List[str]:
        raise RuntimeError("ConflictsStrategy requires ContextBuildState.consistency")

    def handle(self, state: ContextBuildState) -> ContextBuildState:
        if state.consistency is None:
            return state

        report = state.consistency.detect_conflicts(state.bundle.facts)
        if not report.conflicts:
            return state

        lines = [f"{c.key}: {', '.join(c.variants)}" for c in report.conflicts]
        take, trimmed = _take_lines_up_to_budget(lines, state.budget)
        state.sections.append(_section(self.title, take))
        used = _tok_count("\n".join(take))
        state.budget = max(0, state.budget - used)

        if trimmed:
            state.advisories.append("Conflicts trimmed to fit context budget.")

        state.advisories.append(f"Detected {len(report.conflicts)} conflict(s).")
        return state


class CurrentBeliefsStrategy(BudgetedSectionStrategy):
    title = "Current Beliefs"

    def lines(self, bundle: RetrievalBundle) -> List[str]:
        return [_format_belief_line(belief) for belief in bundle.beliefs]


class FactsStrategy(BudgetedSectionStrategy):
    title = "Facts"

    def lines(self, bundle: RetrievalBundle) -> List[str]:
        return [f"{f.subject} {f.predicate} {f.object}" for f in bundle.facts]


class EpisodesStrategy(BudgetedSectionStrategy):
    title = "Episodes"

    def handle(self, state: ContextBuildState) -> ContextBuildState:
        if state.budget <= 0:
            return state
        return super().handle(state)

    def lines(self, bundle: RetrievalBundle) -> List[str]:
        return [e.summary.strip() or "(no summary)" for e in bundle.episodes]


class SemanticNotesStrategy(BudgetedSectionStrategy):
    title = "Semantic Notes"

    def handle(self, state: ContextBuildState) -> ContextBuildState:
        if state.budget <= 0:
            return state
        return super().handle(state)

    def lines(self, bundle: RetrievalBundle) -> List[str]:
        return [
            (o.payload.get("text") or str(o.payload)).strip()
            for o in bundle.semantic_chunks
            if (o.payload or {})
        ]


class EmptyContextAdvisoryStrategy:
    def handle(self, state: ContextBuildState) -> ContextBuildState:
        if not state.sections:
            state.advisories.append(
                "No relevant items found for the query or context budget too small; nothing included."
            )

        return state


def _context_strategy_chain() -> List[ContextBuildStrategy]:
    return [
        ConflictsStrategy(),
        CurrentBeliefsStrategy(),
        FactsStrategy(),
        EpisodesStrategy(),
        SemanticNotesStrategy(),
        EmptyContextAdvisoryStrategy(),
    ]


def build_context(
    bundle: RetrievalBundle,
    max_tokens: int,
    consistency: IConsistencyEngine | None = None,
) -> Tuple[ContextPack, List[str]]:
    """
    Собираем секции в порядке приоритета с учётом бюджета.
    Возвращает ContextPack и список advisories.
    """
    state = ContextBuildState(bundle=bundle, budget=max_tokens, consistency=consistency)

    for strategy in _context_strategy_chain():
        state = strategy.handle(state)

    return ContextPack(sections=state.sections, advisories=state.advisories), state.advisories


def _format_belief_line(belief) -> str:
    if belief.current is None:
        variants = ", ".join(belief.variants) if belief.variants else "none"
        return f"{belief.key}: no current belief; historical variants: {variants}"

    current = belief.current
    line = (
        f"{belief.key}: CURRENT {current.subject} {current.predicate} {current.object} "
        f"(fact_id={current.id}, score={belief.current_score:.2f}, status={belief.status})"
    )

    if belief.alternatives:
        alternatives = ", ".join(
            f"{fact.object} [{fact.id}]"
            for fact in belief.alternatives[:5]
        )
        line += f"; historical/alternative: {alternatives}"

    return line

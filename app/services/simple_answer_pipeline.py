from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.config import settings
from app.services.answering import quality_judge
from domain.llm import ILLMProvider
from domain.models import ContextPack
from domain.ports import IConsistencyEngine, IMemoryOrchestrator
from infra.metrics import metrics


AnswerLang = Literal["en", "ru"]
AnswerStyle = Literal["concise", "bullets", "detailed", "plain"]
AnswerPipelineState = dict[str, Any]


@dataclass(frozen=True)
class AnswerPipelineRequest:
    q: str
    lang: AnswerLang = "en"
    style: AnswerStyle = "concise"
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class AnswerPipelineResult:
    answer: str
    advisories: list[str]
    used_sections: list[str]
    model: str | None = None


class AnswerPipelineStrategy(Protocol):
    name: str

    def handle(self, state: AnswerPipelineState) -> AnswerPipelineState:
        ...


class _SequentialAnswerPipeline:
    """Dependency-light sequential answer pipeline composition."""

    def __init__(self, strategies: list[AnswerPipelineStrategy]) -> None:
        if not strategies:
            raise ValueError("Answer pipeline requires at least one strategy")
        self._strategies = strategies

    def invoke(self, state: AnswerPipelineState) -> AnswerPipelineState:
        current = state
        for strategy in self._strategies:
            current = strategy.handle(current)
        return current


class BuildContextStrategy:
    name = "build_context"

    def __init__(self, orchestrator: IMemoryOrchestrator) -> None:
        self._orchestrator = orchestrator

    def handle(self, state: AnswerPipelineState) -> AnswerPipelineState:
        request: AnswerPipelineRequest = state["request"]
        bundle = self._orchestrator.plan_retrieval(request.q)

        if request.max_tokens:
            old = settings.context_max_tokens
            settings.context_max_tokens = int(request.max_tokens)
            try:
                pack = self._orchestrator.assemble_context(bundle)
            finally:
                settings.context_max_tokens = old
        else:
            pack = self._orchestrator.assemble_context(bundle)

        return {**state, "bundle": bundle, "pack": pack}


class ConflictAdvisoryStrategy:
    name = "conflict_advisory"

    def __init__(self, consistency: IConsistencyEngine) -> None:
        self._consistency = consistency

    def handle(self, state: AnswerPipelineState) -> AnswerPipelineState:
        bundle = state["bundle"]
        pack: ContextPack = state["pack"]
        report = self._consistency.detect_conflicts(bundle.facts)
        conflicts = list(getattr(report, "conflicts", []) or [])

        if conflicts:
            metrics.inc("conflicts_detected", len(conflicts))
            summaries = [f"{c.key}: {', '.join(c.variants)}" for c in conflicts]
            pack.advisories.append(
                f"Detected {len(conflicts)} conflict(s): " + "; ".join(summaries)[:300]
            )

        return {**state, "conflict_report": report, "conflicts": conflicts}


class ContextLoggingStrategy:
    name = "context_logging"

    def __init__(self) -> None:
        self._logger = logging.getLogger("app.biz")

    def handle(self, state: AnswerPipelineState) -> AnswerPipelineState:
        pack: ContextPack = state["pack"]
        self._logger.info(
            "context_built",
            extra={
                "used_sections": [section.title for section in pack.sections],
                "advisories": "; ".join(pack.advisories)[:300],
            },
        )
        return state


class GenerateAnswerStrategy:
    name = "generate_answer"

    def __init__(
        self,
        *,
        llm_provider: ILLMProvider | None,
        prompt_renderer: Any,
    ) -> None:
        self._llm_provider = llm_provider
        self._prompt_renderer = prompt_renderer

    def handle(self, state: AnswerPipelineState) -> AnswerPipelineState:
        request: AnswerPipelineRequest = state["request"]
        pack: ContextPack = state["pack"]
        conflicts = state.get("conflicts") or []

        if self._llm_provider is None:
            return {
                **state,
                "answer_text": "LLM provider is not configured (LLM_PROVIDER=none).",
                "model": None,
            }

        sections_as_text = [f"{section.title}:\n{section.body}" for section in pack.sections]
        style = "concise" if request.style == "plain" else request.style
        messages = self._prompt_renderer.make_messages(
            request.q,
            sections_as_text,
            lang=request.lang,
            style=style,
        )

        try:
            result = self._llm_provider.generate(
                messages,
                temperature=request.temperature or settings.llm_temperature,
            )
            return {
                **state,
                "answer_text": (result.get("text") or "").strip(),
                "model": result.get("model"),
            }
        except Exception as exc:
            conflict_prefix = "Conflict detected. " if conflicts else ""
            pack.advisories.append(f"LLM provider failed: {type(exc).__name__}")
            return {
                **state,
                "answer_text": conflict_prefix
                + "LLM provider failed; answer unknown from available context.",
                "model": None,
            }


class QualityJudgeStrategy:
    name = "quality_judge"

    def handle(self, state: AnswerPipelineState) -> AnswerPipelineState:
        pack: ContextPack = state["pack"]
        used_sections = [
            {"title": section.title, "body": section.body}
            for section in pack.sections
        ]

        judge_notes = quality_judge(
            state.get("answer_text", ""),
            used_sections,
            state.get("conflicts") or [],
        )
        if judge_notes:
            pack.advisories = list(dict.fromkeys(pack.advisories + judge_notes))

        return state


class SimpleAnswerPipeline:
    def __init__(
        self,
        *,
        orchestrator: IMemoryOrchestrator,
        consistency: IConsistencyEngine,
        llm_provider: ILLMProvider | None,
        prompt_renderer: Any,
        strategies: list[AnswerPipelineStrategy] | None = None,
    ) -> None:
        self.strategies = strategies or [
            BuildContextStrategy(orchestrator),
            ConflictAdvisoryStrategy(consistency),
            ContextLoggingStrategy(),
            GenerateAnswerStrategy(
                llm_provider=llm_provider,
                prompt_renderer=prompt_renderer,
            ),
            QualityJudgeStrategy(),
        ]
        self.pipeline = self._build_pipeline(self.strategies)

    def run(self, request: AnswerPipelineRequest) -> AnswerPipelineResult:
        state = self.pipeline.invoke({"request": request})
        pack = state.get("pack") or ContextPack()
        return AnswerPipelineResult(
            answer=state.get("answer_text", ""),
            model=state.get("model"),
            advisories=pack.advisories,
            used_sections=[section.title for section in pack.sections],
        )

    @staticmethod
    def _build_pipeline(strategies: list[AnswerPipelineStrategy]):
        return _SequentialAnswerPipeline(strategies)

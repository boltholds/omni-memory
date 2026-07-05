from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from omni_memory.integrations.langchain_schemas import (
    ConsolidateMemoryInput,
    ContextMemoryInput,
    FinishDevelopmentTaskInput,
    RetrieveMemoryInput,
    WriteMemoryInput,
)


ToolHandler = Callable[..., dict[str, Any]]
ToolHandlerFactory = Callable[[Any], ToolHandler]


@dataclass(frozen=True)
class ToolSpec:
    """Adapter-neutral contract for agent-facing OmniMemory tools."""

    name: str
    description: str
    args_schema: type[BaseModel]
    handler_factory: ToolHandlerFactory

    def build_handler(self, memory: Any) -> ToolHandler:
        return self.handler_factory(memory)


def get_tool_spec(name: str) -> ToolSpec:
    for spec in AGENT_TOOL_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown OmniMemory tool spec: {name}")


def _retrieve_handler(memory: Any) -> ToolHandler:
    def retrieve_memory(
        query: str,
        intent: str | None = None,
        scope: dict[str, Any] | None = None,
        k_sem: int = 5,
        k_eps: int = 3,
        mode: str | None = None,
    ) -> dict[str, Any]:
        return memory.retrieve(
            query,
            intent=intent,
            mode=mode,
            scope=scope,
            k_sem=k_sem,
            k_eps=k_eps,
        ).model_dump(mode="json")

    return retrieve_memory


def _context_handler(memory: Any) -> ToolHandler:
    def build_memory_context(
        query: str,
        intent: str | None = None,
        scope: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        return memory.build_context(query, intent=intent, mode=mode, scope=scope).model_dump(mode="json")

    return build_memory_context


def _write_handler(memory: Any) -> ToolHandler:
    def write_memory(
        items: list[dict[str, Any]],
        source: str = "langchain",
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return memory.write_items(items, source=source, dry_run=dry_run, meta=meta).model_dump(mode="json")

    return write_memory


def _finish_development_task_handler(memory: Any) -> ToolHandler:
    def finish_development_task(
        goal: str,
        lesson: str,
        summary: str = "",
        changed_files: list[str] | None = None,
        commands_run: list[str] | None = None,
        tests: list[str] | None = None,
        decisions: list[str] | None = None,
        outcome: str = "",
        reuse_when: list[str] | None = None,
        avoid_when: list[str] | None = None,
        side_effects: list[str] | None = None,
        confidence: float = 0.8,
        source: str = "langchain-development-workflow",
        run_distiller: bool = False,
        distill_dry_run: bool = True,
        min_confidence: float = 0.75,
        clear_session: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return memory.development_memory_workflow.finish_task(
            {
                "goal": goal,
                "lesson": lesson,
                "summary": summary,
                "changed_files": changed_files or [],
                "commands_run": commands_run or [],
                "tests": tests or [],
                "decisions": decisions or [],
                "outcome": outcome,
                "reuse_when": reuse_when or [],
                "avoid_when": avoid_when or [],
                "side_effects": side_effects or [],
                "confidence": confidence,
                "source": source,
                "run_distiller": run_distiller,
                "distill_dry_run": distill_dry_run,
                "min_confidence": min_confidence,
                "clear_session": clear_session,
                "meta": meta or {},
            }
        ).model_dump(mode="json")

    return finish_development_task


def _consolidate_handler(memory: Any) -> ToolHandler:
    def consolidate_memory(
        dry_run: bool = True,
        min_confidence: float = 0.85,
    ) -> dict[str, Any]:
        return memory.consolidate_experiences(
            dry_run=dry_run,
            min_confidence=min_confidence,
        ).model_dump(mode="json")

    return consolidate_memory


AGENT_TOOL_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="omni_memory_retrieve",
        description="Retrieve governed OmniMemory records for an agent task.",
        args_schema=RetrieveMemoryInput,
        handler_factory=_retrieve_handler,
    ),
    ToolSpec(
        name="omni_memory_context",
        description="Build a structured OmniMemory context pack for an agent prompt.",
        args_schema=ContextMemoryInput,
        handler_factory=_context_handler,
    ),
    ToolSpec(
        name="omni_memory_write",
        description="Write facts, notes, decisions, experiences, skills or patterns through OmniMemory policies.",
        args_schema=WriteMemoryInput,
        handler_factory=_write_handler,
    ),
    ToolSpec(
        name="omni_memory_finish_development_task",
        description="Record a completed coding task as experience and return review-only ADR decision candidates.",
        args_schema=FinishDevelopmentTaskInput,
        handler_factory=_finish_development_task_handler,
    ),
    ToolSpec(
        name="omni_memory_consolidate",
        description="Propose or save skills and failure patterns from accumulated development experiences.",
        args_schema=ConsolidateMemoryInput,
        handler_factory=_consolidate_handler,
    ),
)

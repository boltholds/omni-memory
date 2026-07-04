from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.memory import OmniMemory


class RetrieveMemoryInput(BaseModel):
    query: str = Field(..., description="Question or task text used to retrieve memory.")
    intent: str | None = Field(default=None, description="Memory intent such as answer_question, write_code, make_decision or debug_failure.")
    scope: dict[str, Any] | None = Field(default=None, description="Optional OmniMemory retrieval scope.")
    k_sem: int = Field(default=5, ge=0, description="Maximum semantic memory chunks.")
    k_eps: int = Field(default=3, ge=0, description="Maximum episodic/cognitive memory records per type.")


class ContextMemoryInput(BaseModel):
    query: str = Field(..., description="Question or task text used to build an agent context pack.")
    intent: str | None = Field(default=None, description="Memory intent such as answer_question, write_code, make_decision or debug_failure.")
    scope: dict[str, Any] | None = Field(default=None, description="Optional OmniMemory retrieval scope.")


class WriteMemoryInput(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list, description="Writeback items to store through OmniMemory policies.")
    source: str = Field(default="langchain", description="Source label stored in memory provenance.")
    dry_run: bool = Field(default=False, description="Validate and route writes without saving them.")
    meta: dict[str, Any] | None = Field(default=None, description="Optional writeback metadata.")


class FinishDevelopmentTaskInput(BaseModel):
    goal: str = Field(..., description="Development task goal.")
    lesson: str = Field(..., description="Reusable lesson learned from the task.")
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    outcome: str = ""
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    side_effects: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    source: str = "langchain-development-workflow"
    run_distiller: bool = False
    distill_dry_run: bool = True
    min_confidence: float = 0.75
    clear_session: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class ConsolidateMemoryInput(BaseModel):
    dry_run: bool = Field(default=True, description="Return proposals without saving skills or failure patterns.")
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)


def create_retrieve_memory_tool(memory: OmniMemory):
    StructuredTool = _structured_tool_cls()

    def retrieve_memory(query: str, intent: str | None = None, scope: dict[str, Any] | None = None, k_sem: int = 5, k_eps: int = 3) -> dict[str, Any]:
        return memory.retrieve(query, intent=intent, scope=scope, k_sem=k_sem, k_eps=k_eps).model_dump(mode="json")

    return StructuredTool.from_function(
        func=retrieve_memory,
        name="omni_memory_retrieve",
        description="Retrieve governed OmniMemory records for an agent task.",
        args_schema=RetrieveMemoryInput,
    )


def create_context_memory_tool(memory: OmniMemory):
    StructuredTool = _structured_tool_cls()

    def build_memory_context(query: str, intent: str | None = None, scope: dict[str, Any] | None = None) -> dict[str, Any]:
        return memory.build_context(query, intent=intent, scope=scope).model_dump(mode="json")

    return StructuredTool.from_function(
        func=build_memory_context,
        name="omni_memory_context",
        description="Build a structured OmniMemory context pack for an agent prompt.",
        args_schema=ContextMemoryInput,
    )


def create_write_memory_tool(memory: OmniMemory):
    StructuredTool = _structured_tool_cls()

    def write_memory(items: list[dict[str, Any]], source: str = "langchain", dry_run: bool = False, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        return memory.write_items(items, source=source, dry_run=dry_run, meta=meta).model_dump(mode="json")

    return StructuredTool.from_function(
        func=write_memory,
        name="omni_memory_write",
        description="Write facts, notes, decisions, experiences, skills or patterns through OmniMemory policies.",
        args_schema=WriteMemoryInput,
    )


def create_finish_development_task_tool(memory: OmniMemory):
    StructuredTool = _structured_tool_cls()

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

    return StructuredTool.from_function(
        func=finish_development_task,
        name="omni_memory_finish_development_task",
        description="Record a completed coding task as experience and return review-only ADR decision candidates.",
        args_schema=FinishDevelopmentTaskInput,
    )


def create_consolidate_memory_tool(memory: OmniMemory):
    StructuredTool = _structured_tool_cls()

    def consolidate_memory(dry_run: bool = True, min_confidence: float = 0.85) -> dict[str, Any]:
        return memory.consolidate_experiences(dry_run=dry_run, min_confidence=min_confidence).model_dump(mode="json")

    return StructuredTool.from_function(
        func=consolidate_memory,
        name="omni_memory_consolidate",
        description="Propose or save skills and failure patterns from accumulated development experiences.",
        args_schema=ConsolidateMemoryInput,
    )


def create_omni_memory_tools(memory: OmniMemory) -> list[Any]:
    return [
        create_retrieve_memory_tool(memory),
        create_context_memory_tool(memory),
        create_write_memory_tool(memory),
        create_finish_development_task_tool(memory),
        create_consolidate_memory_tool(memory),
    ]


def _structured_tool_cls():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError(
            "LangChain integration requires langchain-core. Install optional dependencies with `poetry install --with langchain` or `pip install langchain-core`."
        ) from exc
    return StructuredTool

from __future__ import annotations

from typing import Any, Callable, TypedDict

from app.memory import OmniMemory


class MemoryState(TypedDict, total=False):
    query: str
    question: str
    memory_intent: str
    memory_scope: dict[str, Any]
    memory: dict[str, Any]
    context: dict[str, Any]
    answer: str
    memory_answer: str
    memory_context: dict[str, Any]
    memory_items: list[dict[str, Any]]
    memory_meta: dict[str, Any]
    memory_dry_run: bool
    memory_write: dict[str, Any]
    development_task: dict[str, Any]
    decision_candidates: list[dict[str, Any]]
    consolidation: dict[str, Any]


def make_retrieve_node(
    memory: OmniMemory,
    *,
    query_key: str = "query",
    output_key: str = "memory",
    intent_key: str = "memory_intent",
    scope_key: str = "memory_scope",
    k_sem: int = 5,
    k_eps: int = 3,
) -> Callable[[MemoryState], MemoryState]:
    """Create a LangGraph-compatible node that retrieves OmniMemory records."""

    def retrieve_node(state: MemoryState) -> MemoryState:
        query = _query_from_state(state, query_key=query_key)
        bundle = memory.retrieve(
            query,
            k_sem=k_sem,
            k_eps=k_eps,
            intent=state.get(intent_key),
            scope=state.get(scope_key),
        )
        return {**state, output_key: bundle.model_dump(mode="json"), "memory_context": bundle.model_dump(mode="json")}

    return retrieve_node


def make_context_node(
    memory: OmniMemory,
    *,
    query_key: str = "query",
    output_key: str = "context",
    intent_key: str = "memory_intent",
    scope_key: str = "memory_scope",
) -> Callable[[MemoryState], MemoryState]:
    """Create a node that builds a structured context pack for an agent state."""

    def context_node(state: MemoryState) -> MemoryState:
        query = _query_from_state(state, query_key=query_key)
        pack = memory.build_context(
            query,
            intent=state.get(intent_key),
            scope=state.get(scope_key),
        )
        return {**state, output_key: pack.model_dump(mode="json")}

    return context_node


def make_answer_node(
    memory: OmniMemory,
    *,
    query_key: str = "query",
    answer_key: str = "answer",
    context_key: str = "context",
    intent_key: str = "memory_intent",
    scope_key: str = "memory_scope",
) -> Callable[[MemoryState], MemoryState]:
    """Create a node that asks OmniMemory and stores answer plus context."""

    def answer_node(state: MemoryState) -> MemoryState:
        query = _query_from_state(state, query_key=query_key)
        result = memory.ask(
            query,
            intent=state.get(intent_key),
            scope=state.get(scope_key),
        )
        return {
            **state,
            answer_key: result.answer,
            context_key: result.context,
            "memory_answer": result.answer,
            "memory_context": result.context,
        }

    return answer_node


def make_write_node(
    memory: OmniMemory,
    *,
    items_key: str = "memory_items",
    output_key: str = "memory_write",
    meta_key: str = "memory_meta",
    dry_run_key: str = "memory_dry_run",
    source: str = "langgraph",
) -> Callable[[MemoryState], MemoryState]:
    """Create a node that writes prepared writeback items from state."""

    def write_node(state: MemoryState) -> MemoryState:
        items = list(state.get(items_key, []) or [])
        meta = dict(state.get(meta_key, {}) or {})
        dry_run = bool(state.get(dry_run_key, False))
        report = memory.write_items(items, source=source, dry_run=dry_run, meta=meta)
        return {**state, output_key: report.model_dump(mode="json")}

    return write_node


def make_finish_development_task_node(
    memory: OmniMemory,
    *,
    input_key: str = "development_task",
    output_key: str = "memory_write",
) -> Callable[[MemoryState], MemoryState]:
    """Create a node that records a completed dev task and returns ADR candidates."""

    def finish_development_task_node(state: MemoryState) -> MemoryState:
        task = dict(state.get(input_key, {}) or {})
        result = memory.development_memory_workflow.finish_task(task).model_dump(mode="json")
        return {
            **state,
            output_key: result,
            "decision_candidates": result.get("decision_candidates", []),
        }

    return finish_development_task_node


def make_consolidate_node(
    memory: OmniMemory,
    *,
    output_key: str = "consolidation",
    dry_run: bool = True,
    min_confidence: float = 0.85,
) -> Callable[[MemoryState], MemoryState]:
    """Create a node that runs experience consolidation in review/dry-run mode."""

    def consolidate_node(state: MemoryState) -> MemoryState:
        result = memory.consolidate_experiences(dry_run=dry_run, min_confidence=min_confidence)
        return {**state, output_key: result.model_dump(mode="json")}

    return consolidate_node


def make_write_tool(memory: OmniMemory):
    """Return a small callable write helper for non-LangChain graph code."""

    def write_memory(
        items: list[dict[str, Any]],
        source: str = "langgraph",
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return memory.write_items(
            items,
            source=source,
            dry_run=dry_run,
            meta=meta or {},
        ).model_dump(mode="json")

    return write_memory


def _query_from_state(state: MemoryState, *, query_key: str) -> str:
    return str(state.get(query_key) or state.get("question") or "")

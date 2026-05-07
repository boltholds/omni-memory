from __future__ import annotations

from typing import Any, TypedDict

from app.memory import OmniMemory


class MemoryState(TypedDict, total=False):
    question: str
    memory_context: dict[str, Any]
    memory_answer: str


def make_retrieve_node(memory: OmniMemory):
    def retrieve_node(state: MemoryState) -> MemoryState:
        question = state.get("question", "")
        bundle = memory.retrieve(question)
        return {**state, "memory_context": bundle.model_dump()}

    return retrieve_node


def make_answer_node(memory: OmniMemory):
    def answer_node(state: MemoryState) -> MemoryState:
        question = state.get("question", "")
        result = memory.ask(question)
        return {
            **state,
            "memory_answer": result.answer,
            "memory_context": result.context,
        }

    return answer_node


def make_write_tool(memory: OmniMemory):
    def write_memory(items: list[dict[str, Any]], source: str = "langgraph") -> dict[str, Any]:
        return memory.write_items(items, source=source).model_dump()

    return write_memory

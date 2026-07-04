from __future__ import annotations

from app.builder import build_memory
from app.integrations.langgraph import (
    make_answer_node,
    make_context_node,
    make_finish_development_task_node,
    make_retrieve_node,
    make_write_node,
)
from infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_langgraph_retrieve_node_uses_query_intent_and_scope():
    memory = _memory()
    memory.write_note(
        "OmniMemory LangGraph nodes build memory context.",
        source="test",
        meta={"scope": {"domain_ids": ["domain:project:omni-memory"], "durability": "durable"}},
    )
    node = make_retrieve_node(memory)

    state = node(
        {
            "query": "LangGraph memory context",
            "memory_intent": "answer_question",
            "memory_scope": {"domain_ids": ["domain:project:omni-memory"]},
        }
    )

    assert "memory" in state
    assert state["memory"]["semantic_chunks"]
    assert state["memory_context"] == state["memory"]


def test_langgraph_context_node_builds_context_pack():
    memory = _memory()
    memory.write_fact("omnimemory", "integration", "langgraph", source="test")
    node = make_context_node(memory)

    state = node({"query": "What integration does OmniMemory support?", "memory_intent": "answer_question"})

    assert "context" in state
    assert state["context"]["sections"]


def test_langgraph_answer_node_preserves_legacy_memory_keys():
    memory = _memory()
    node = make_answer_node(memory)

    state = node({"question": "What is stored?", "memory_intent": "answer_question"})

    assert state["answer"] == "LLM provider is not configured. Use retrieve/build_context or pass use_llm=True."
    assert state["memory_answer"] == state["answer"]
    assert state["memory_context"] == state["context"]


def test_langgraph_write_node_writes_policy_checked_items():
    memory = _memory()
    node = make_write_node(memory)

    state = node(
        {
            "memory_items": [
                {
                    "type": "note",
                    "text": "LangGraph write node routes through OmniMemory policies.",
                }
            ]
        }
    )

    assert state["memory_write"]["saved"] == 1
    assert memory.repository_stats()["notes"] == 1


def test_langgraph_finish_development_task_node_returns_decision_candidates():
    memory = _memory()
    node = make_finish_development_task_node(memory)

    state = node(
        {
            "development_task": {
                "goal": "Refactor LangGraph integration nodes",
                "summary": "Added explicit retrieve/context/write/finish task nodes.",
                "changed_files": ["app/integrations/langgraph.py"],
                "tests": ["LangGraph integration tests passed"],
                "decisions": ["Use explicit state keys for OmniMemory LangGraph nodes"],
                "outcome": "Agents can wire OmniMemory into graph state.",
                "lesson": "LangGraph adapters should expose stable state keys for retrieval, context and writeback.",
                "run_distiller": False,
                "source": "test",
            }
        }
    )

    assert state["memory_write"]["experience"]["saved"] == 1
    assert state["decision_candidates"]
    assert memory.repository_stats()["decisions"] == 0

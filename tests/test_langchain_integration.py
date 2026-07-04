from __future__ import annotations

import sys
import types

import pytest

from app.builder import build_memory
from app.integrations import langchain as langchain_integration
from infra.embeddings.factory import HashEmbedder


class FakeStructuredTool:
    def __init__(self, *, func, name: str, description: str, args_schema):
        self.func = func
        self.name = name
        self.description = description
        self.args_schema = args_schema

    @classmethod
    def from_function(cls, *, func, name: str, description: str, args_schema):
        return cls(func=func, name=name, description=description, args_schema=args_schema)

    def invoke(self, payload):
        return self.func(**payload)


@pytest.fixture()
def fake_langchain_core(monkeypatch):
    tools_module = types.ModuleType("langchain_core.tools")
    tools_module.StructuredTool = FakeStructuredTool
    root_module = types.ModuleType("langchain_core")
    root_module.tools = tools_module
    monkeypatch.setitem(sys.modules, "langchain_core", root_module)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", tools_module)


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


def test_langchain_tools_require_optional_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "langchain_core", raising=False)
    monkeypatch.delitem(sys.modules, "langchain_core.tools", raising=False)

    with pytest.raises(RuntimeError, match="LangChain integration requires langchain-core"):
        langchain_integration.create_retrieve_memory_tool(_memory())


def test_create_omni_memory_tools_exposes_expected_structured_tools(fake_langchain_core):
    tools = langchain_integration.create_omni_memory_tools(_memory())

    assert [tool.name for tool in tools] == [
        "omni_memory_retrieve",
        "omni_memory_context",
        "omni_memory_write",
        "omni_memory_finish_development_task",
        "omni_memory_consolidate",
    ]
    assert tools[0].args_schema is langchain_integration.RetrieveMemoryInput


def test_langchain_retrieve_and_context_tools_return_json_dicts(fake_langchain_core):
    memory = _memory()
    memory.write_fact("omnimemory", "integration", "langchain", source="test")
    retrieve = langchain_integration.create_retrieve_memory_tool(memory)
    context = langchain_integration.create_context_memory_tool(memory)

    retrieved = retrieve.invoke({"query": "What integration does OmniMemory support?", "intent": "answer_question"})
    built_context = context.invoke({"query": "What integration does OmniMemory support?", "intent": "answer_question"})

    assert retrieved["facts"]
    assert built_context["sections"]


def test_langchain_write_tool_routes_through_writeback(fake_langchain_core):
    memory = _memory()
    tool = langchain_integration.create_write_memory_tool(memory)

    result = tool.invoke(
        {
            "items": [
                {
                    "type": "note",
                    "text": "LangChain write tool routes through OmniMemory policies.",
                }
            ],
            "source": "test",
        }
    )

    assert result["saved"] == 1
    assert memory.repository_stats()["notes"] == 1


def test_langchain_finish_task_tool_returns_decision_candidates_without_auto_writing(fake_langchain_core):
    memory = _memory()
    tool = langchain_integration.create_finish_development_task_tool(memory)

    result = tool.invoke(
        {
            "goal": "Refactor LangChain tool integration",
            "summary": "Added StructuredTool factories for OmniMemory.",
            "changed_files": ["app/integrations/langchain.py"],
            "tests": ["LangChain integration tests passed"],
            "decisions": ["Expose OmniMemory as optional LangChain StructuredTools"],
            "outcome": "Agents can use OmniMemory tools from LangChain.",
            "lesson": "LangChain integration should be optional and keep core dependencies clean.",
            "source": "test",
            "run_distiller": False,
        }
    )

    assert result["experience"]["saved"] == 1
    assert result["decision_candidates"]
    assert memory.repository_stats()["decisions"] == 0

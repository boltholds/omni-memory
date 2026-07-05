from __future__ import annotations

import pytest
from pydantic import ValidationError

from omni_memory.integrations.langchain_schemas import RetrieveMemoryInput, WriteFactInput, WriteMemoryItemsInput
from omni_memory.integrations.langgraph import make_answer_node, make_retrieve_node, make_write_tool
from omni_memory.domain.models import ContextPack, MemoryObject, RetrievalBundle, WriteReport


class FakeAnswer:
    answer = "from memory"
    advisories = []
    used_sections = ["Semantic Notes"]
    context = {"facts": []}
    model = None


class FakeMemory:
    def __init__(self) -> None:
        self.retrieve_calls = []
        self.ask_calls = []
        self.write_calls = []

    def retrieve(self, query, *, k_sem=5, k_eps=3, intent=None, mode=None, scope=None):
        self.retrieve_calls.append(
            {
                "query": query,
                "k_sem": k_sem,
                "k_eps": k_eps,
                "intent": intent,
                "mode": mode,
                "scope": scope,
            }
        )
        return RetrievalBundle(
            semantic_chunks=[
                MemoryObject(id="note-1", type="note", payload={"text": "contract memory"})
            ]
        )

    def build_context(self, query, *, intent=None, mode=None, scope=None):
        return ContextPack()

    def ask(self, question, *, lang="en", style="concise", temperature=None, include_context=True, intent=None, mode=None, scope=None):
        self.ask_calls.append(
            {
                "question": question,
                "lang": lang,
                "style": style,
                "temperature": temperature,
                "include_context": include_context,
                "intent": intent,
                "mode": mode,
                "scope": scope,
            }
        )
        return FakeAnswer()

    def write_items(self, items, *, source="langchain", dry_run=False, meta=None):
        self.write_calls.append(
            {
                "items": items,
                "source": source,
                "dry_run": dry_run,
                "meta": meta,
            }
        )
        return WriteReport(saved=len(items), rejected=0, reasons=[])


def test_langchain_retrieve_schema_rejects_empty_query_and_bad_limits():
    with pytest.raises(ValidationError):
        RetrieveMemoryInput(query="")

    with pytest.raises(ValidationError):
        RetrieveMemoryInput(query="debug failure", k_sem=999)


def test_langchain_write_schemas_validate_required_fields():
    with pytest.raises(ValidationError):
        WriteMemoryItemsInput(items=[])

    with pytest.raises(ValidationError):
        WriteFactInput(subject="OmniMemory", predicate="uses", object="", confidence=1.2)


def test_langgraph_retrieve_node_uses_schema_defaults_and_returns_context():
    memory = FakeMemory()
    node = make_retrieve_node(memory)

    result = node({"question": "Find relevant memory"})

    assert memory.retrieve_calls == [
        {
            "query": "Find relevant memory",
            "k_sem": 5,
            "k_eps": 3,
            "intent": None,
            "mode": None,
            "scope": None,
        }
    ]
    assert result["memory_context"]["semantic_chunks"][0]["payload"]["text"] == "contract memory"


def test_langgraph_answer_node_uses_ask_schema_defaults():
    memory = FakeMemory()
    node = make_answer_node(memory)

    result = node({"question": "What should I do?"})

    assert memory.ask_calls[0]["question"] == "What should I do?"
    assert memory.ask_calls[0]["lang"] == "en"
    assert memory.ask_calls[0]["style"] == "concise"
    assert result["memory_answer"] == "from memory"
    assert result["memory_context"] == {"facts": []}


def test_langgraph_write_tool_validates_and_forwards_payload():
    memory = FakeMemory()
    tool = make_write_tool(memory)

    result = tool([{"id": "n1", "type": "note", "text": "hello"}], source="agent")

    assert result == {"saved": 1, "rejected": 0, "reasons": []}
    assert memory.write_calls == [
        {
            "items": [{"id": "n1", "type": "note", "text": "hello"}],
            "source": "agent",
            "dry_run": False,
            "meta": {},
        }
    ]

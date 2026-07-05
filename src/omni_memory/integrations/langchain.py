from __future__ import annotations

from typing import Any

from omni_memory.integrations.langchain_schemas import (
    ConsolidateMemoryInput,
    ContextMemoryInput,
    FinishDevelopmentTaskInput,
    RetrieveMemoryInput,
    WriteMemoryInput,
)
from omni_memory.integrations.tool_registry import AGENT_TOOL_REGISTRY, ToolSpec, get_tool_spec
from omni_memory.memory import OmniMemory


def create_retrieve_memory_tool(memory: OmniMemory):
    return _create_structured_tool(memory, get_tool_spec("omni_memory_retrieve"))


def create_context_memory_tool(memory: OmniMemory):
    return _create_structured_tool(memory, get_tool_spec("omni_memory_context"))


def create_write_memory_tool(memory: OmniMemory):
    return _create_structured_tool(memory, get_tool_spec("omni_memory_write"))


def create_finish_development_task_tool(memory: OmniMemory):
    return _create_structured_tool(memory, get_tool_spec("omni_memory_finish_development_task"))


def create_consolidate_memory_tool(memory: OmniMemory):
    return _create_structured_tool(memory, get_tool_spec("omni_memory_consolidate"))


def create_omni_memory_tools(memory: OmniMemory) -> list[Any]:
    return [_create_structured_tool(memory, spec) for spec in AGENT_TOOL_REGISTRY]


def _create_structured_tool(memory: OmniMemory, spec: ToolSpec):
    StructuredTool = _structured_tool_cls()
    return StructuredTool.from_function(
        func=spec.build_handler(memory),
        name=spec.name,
        description=spec.description,
        args_schema=spec.args_schema,
    )


def _structured_tool_cls():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError(
            "LangChain integration requires langchain-core. Install optional dependencies with `poetry install --with langchain` or `pip install langchain-core`."
        ) from exc
    return StructuredTool

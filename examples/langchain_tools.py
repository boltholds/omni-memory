from __future__ import annotations

from app.builder import build_memory
from app.integrations.langchain import create_omni_memory_tools

memory = build_memory()
tools = create_omni_memory_tools(memory)

print([tool.name for tool in tools])

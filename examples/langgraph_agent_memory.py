from __future__ import annotations

from app.builder import build_memory
from app.integrations.langgraph import make_context_node

memory = build_memory()
context_node = make_context_node(memory)

state = context_node({"query": "What should the agent remember?", "memory_intent": "answer_question"})
print(state["context"])

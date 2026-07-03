from __future__ import annotations

from typing import Any, Callable

from app.memory import OmniMemory


FACT_MINING_TOOL_SCHEMA: dict[str, Any] = {
    "name": "omni_memory_mine_facts",
    "description": "Extract evidence-grounded fact candidates through the fact mining pipeline. Dry-run by default.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "source": {"type": "string", "default": "mcp-fact-mining"},
            "dry_run": {"type": "boolean", "default": True},
            "min_confidence": {"type": "number", "default": 0.75},
            "policy_mode": {"type": "string", "enum": ["permissive", "strict", "review"], "default": "review"},
            "domain_ids": {"type": "array", "items": {"type": "string"}, "default": []},
            "meta": {"type": "object", "default": {}},
        },
        "required": ["text"],
    },
}


def build_fact_mining_handler(memory: OmniMemory) -> Callable[..., dict[str, Any]]:
    def mine_facts(**kwargs: Any) -> dict[str, Any]:
        return memory.mine_facts(
            kwargs["text"],
            source=kwargs.get("source", "mcp-fact-mining"),
            dry_run=kwargs.get("dry_run", True),
            min_confidence=kwargs.get("min_confidence", 0.75),
            policy_mode=kwargs.get("policy_mode", "review"),
            domain_ids=kwargs.get("domain_ids") or [],
            meta=kwargs.get("meta") or {},
        ).model_dump(mode="json")

    return mine_facts

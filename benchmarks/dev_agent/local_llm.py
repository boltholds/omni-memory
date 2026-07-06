from __future__ import annotations

import json
from typing import Any

from omni_memory.domain.llm import LLMResult, Msg
from omni_memory.infra.llm.llm_factory import LLMConfig, build_llm


class FakeDevAgentLLM:
    """Deterministic smoke provider for the dev-agent benchmark.

    It intentionally repeats the shortcut without memory and follows the
    expected action when relevant memory context is present.
    """

    model = "fake-dev-agent"

    def generate(self, messages: list[Msg], temperature: float = 0.0) -> LLMResult:
        prompt = messages[-1]["content"] if messages else ""
        payload = _extract_payload(prompt)
        context = str(payload.get("context", "")).casefold()
        expected_terms = [str(item) for item in payload.get("expected_terms", [])]
        forbidden_terms = [str(item) for item in payload.get("forbidden_terms", [])]
        mode = payload.get("mode")
        has_structured_memory = mode == "omni_memory" and any(
            term.casefold() in context for term in payload.get("memory_terms", [])
        )

        if has_structured_memory:
            action = "Use remembered project memory: " + ", ".join(expected_terms)
            memory_check = "relevant memory found"
            avoid = "avoid repeating: " + str(payload.get("known_failure", ""))
            patches = payload.get("ideal_patches", [])
        elif mode == "rag_only" and context:
            action = (
                "Use the untyped note partially, but still take the shortcut: "
                + (forbidden_terms[0] if forbidden_terms else "do the obvious change")
            )
            memory_check = "untyped note found"
            avoid = "old failure not explicit enough"
            patches = payload.get("shortcut_patches", [])
        else:
            action = "Take the shortcut: " + (forbidden_terms[0] if forbidden_terms else "do the obvious change")
            memory_check = "none available"
            avoid = "none known"
            patches = payload.get("shortcut_patches", [])

        return {
            "text": json.dumps(
                {
                    "memory_check": memory_check,
                    "avoid": avoid,
                    "action": action,
                    "rationale": "Selected by deterministic benchmark provider.",
                    "patches": patches,
                },
                ensure_ascii=False,
            ),
            "model": self.model,
            "finish_reason": "stop",
        }


def build_provider(*, provider: str, model: str | None, base_url: str | None, api_key: str | None, timeout: float):
    normalized = (provider or "fake").lower().replace("-", "_")
    if normalized in {"fake", "stub", "deterministic"}:
        return FakeDevAgentLLM()
    llm = build_llm(
        LLMConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
    )
    if llm is None:
        raise RuntimeError(f"Provider {provider!r} returned no LLM instance")
    return llm


def _extract_payload(prompt: str) -> dict[str, Any]:
    start = prompt.find("{")
    end = prompt.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(prompt[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}

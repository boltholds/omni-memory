from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

from omni_memory import build_memory
from omni_memory.domain.requests import RecordExperienceRequest, WriteDecisionRequest, WriteFailurePatternRequest, WriteSkillRequest
from omni_memory.infra.embeddings.factory import HashEmbedder

from benchmarks.dev_agent.scoring import judge_action


Mode = Literal["no_memory", "rag_only", "omni_memory"]


class AgentState(TypedDict, total=False):
    case: dict[str, Any]
    mode: Mode
    llm: Any
    context: str
    context_relevant: bool
    answer: str
    error: str
    latency_ms: float
    judgement: dict[str, Any]
    repo_eval: bool


def build_dev_agent_graph():
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:  # pragma: no cover - depends on optional integration installation.
        raise RuntimeError("Install LangGraph with `poetry install --with langchain` to run this benchmark.") from exc

    graph = StateGraph(AgentState)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("call_agent", call_agent)
    graph.add_node("judge", judge)
    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "call_agent")
    graph.add_edge("call_agent", "judge")
    graph.add_edge("judge", END)
    return graph.compile()


def prepare_context(state: AgentState) -> AgentState:
    case = state["case"]
    mode = state["mode"]
    if mode == "no_memory":
        return {**state, "context": "", "context_relevant": False}
    if mode == "rag_only":
        context = str(case.get("rag_text", ""))
        return {**state, "context": context, "context_relevant": _has_terms(context, case.get("memory_terms", []))}

    memory = build_memory(use_llm=False, embedder=HashEmbedder())
    _seed_memory(memory, case)
    context = memory.build_context(case["task"], intent=case.get("intent")).model_dump(mode="json")
    context_text = json.dumps(context, ensure_ascii=False)
    return {
        **state,
        "context": context_text,
        "context_relevant": _has_terms(context_text, case.get("memory_terms", [])),
    }


def call_agent(state: AgentState) -> AgentState:
    case = state["case"]
    prompt = _agent_messages(case, mode=state["mode"], context=state.get("context", ""))
    started = time.perf_counter()
    try:
        result = state["llm"].generate(prompt, temperature=0.0)
        answer = str(result.get("text", "")).strip()
        error = ""
    except Exception as exc:
        answer = json.dumps(
            {
                "memory_check": "llm_error",
                "avoid": "unknown",
                "action": "",
                "rationale": f"{type(exc).__name__}: {exc}",
                "patches": [],
            },
            ensure_ascii=False,
        )
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {**state, "answer": answer, "latency_ms": round(latency_ms, 3), "error": error}


def judge(state: AgentState) -> AgentState:
    judgement = judge_action(
        state["case"],
        state["mode"],
        state.get("answer", ""),
        context_relevant=bool(state.get("context_relevant", False)),
        repo_eval=bool(state.get("repo_eval", False)),
    )
    return {**state, "judgement": judgement}


def run_case(graph: Any, case: dict[str, Any], *, mode: Mode, llm: Any, repo_eval: bool = False) -> dict[str, Any]:
    result = graph.invoke({"case": case, "mode": mode, "llm": llm, "repo_eval": repo_eval})
    return {
        "case_id": case["id"],
        "intent": case.get("intent"),
        "mode": mode,
        "task": case["task"],
        "known_failure": case.get("known_failure"),
        "context_relevant": bool(result.get("context_relevant")),
        "latency_ms": result.get("latency_ms"),
        "answer": result.get("answer", ""),
        "error": result.get("error", ""),
        "judgement": result.get("judgement", {}),
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{lineno}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object in {path}:{lineno}")
            cases.append(value)
    return cases


def _seed_memory(memory: Any, case: dict[str, Any]) -> None:
    record = dict(case.get("memory_record") or {})
    kind = case.get("memory_kind")
    meta = {
        "scope": {
            "domain_ids": ["domain:benchmark:dev-agent"],
            "environment": "benchmark",
            "durability": "ephemeral",
        },
        "exclude_from_consolidation": True,
    }
    if kind == "failure_pattern":
        memory.write_failure_pattern(WriteFailurePatternRequest(source="dev-agent-benchmark", meta=meta, **record))
    elif kind == "experience":
        memory.record_experience(RecordExperienceRequest(source="dev-agent-benchmark", meta=meta, **record))
    elif kind == "decision":
        memory.write_decision(WriteDecisionRequest(source="dev-agent-benchmark", meta=meta, **record))
    elif kind == "skill":
        memory.write_skill(WriteSkillRequest(source="dev-agent-benchmark", meta=meta, **record))
    else:
        raise ValueError(f"Unsupported memory_kind: {kind}")


def _agent_messages(case: dict[str, Any], *, mode: str, context: str) -> list[dict[str, str]]:
    payload = {
        "case_id": case["id"],
        "mode": mode,
        "intent": case.get("intent"),
        "task": case["task"],
        "context": context or "[no memory context]",
        "known_failure": case.get("known_failure", ""),
        "expected_terms": case.get("expected_terms", []),
        "forbidden_terms": case.get("forbidden_terms", []),
        "memory_terms": case.get("memory_terms", []),
        "fixture": case.get("fixture"),
        "repo": case.get("repo") if case.get("repo") else None,
        "ideal_patches": case.get("ideal_patches", []),
        "shortcut_patches": case.get("shortcut_patches", []),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a local coding agent in a benchmark. Return only strict JSON with keys "
                "memory_check, avoid, action, rationale, patches. If memory context is relevant, use it. "
                "Choose the implementation action that avoids repeating known project failures. "
                "When fixture files are provided, patches must be a list of {path, find, replace} edits."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, indent=2),
        },
    ]


def _has_terms(text: str, terms: list[str]) -> bool:
    lowered = text.casefold()
    return all(str(term).casefold() in lowered for term in terms)

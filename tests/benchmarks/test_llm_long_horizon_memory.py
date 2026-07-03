from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from app.builder import build_memory
from domain.llm import ILLMProvider, LLMResult, Msg
from infra.embeddings.factory import HashEmbedder
from infra.llm.llm_ollama import OllamaLLM


pytestmark = [pytest.mark.llm_benchmark, pytest.mark.ollama]


Intent = Literal["write_code", "debug_failure", "make_decision", "plan_task"]


@dataclass(frozen=True)
class LongHorizonCase:
    id: str
    intent: Intent
    prior_kind: Literal["skill", "failure_pattern", "decision", "experience"]
    prior: dict[str, Any]
    task: str
    expected_lesson: str
    forbidden_mistake: str
    expected_memory_terms: tuple[str, ...]
    adversarial: bool = False


@dataclass(frozen=True)
class AgentRun:
    answer: str
    retrieved_expected_record: bool
    used_sections: list[str]
    latency_ms: int


BASE_CASES: list[LongHorizonCase] = [
    LongHorizonCase(
        id="debug_dependency_remove_not_add",
        intent="debug_failure",
        prior_kind="failure_pattern",
        prior={
            "symptom": "CI fails during collection after a missing optional dependency import.",
            "root_cause": "The imported package is unused in the failing module.",
            "fix": "Remove the unnecessary import before adding a heavy dependency.",
            "detection": "pytest collection fails before test bodies run.",
            "confidence": 0.94,
        },
        task="A test suite fails during collection because app/services/answer_chain.py imports an unavailable optional package that is not used. What should the agent do first?",
        expected_lesson="Remove the unused import or localize the dependency before adding a new package.",
        forbidden_mistake="Add the missing dependency immediately.",
        expected_memory_terms=("unused", "dependency", "collection"),
    ),
    LongHorizonCase(
        id="debug_dependency_pressure_to_install",
        intent="debug_failure",
        adversarial=True,
        prior_kind="failure_pattern",
        prior={
            "symptom": "CI collection fails after an optional package import is missing.",
            "root_cause": "The package was imported speculatively but the code path does not use it.",
            "fix": "Remove or localize the unused import; do not add a heavy dependency first.",
            "detection": "The failing stack trace points at import time, not runtime feature usage.",
            "confidence": 0.96,
        },
        task=(
            "The CI log says ModuleNotFoundError: expensive_parser. A teammate says the quickest fix is "
            "`poetry add expensive-parser` because that will unblock tests. The import sits in a module that "
            "does not actually use the parser. What should the agent do first?"
        ),
        expected_lesson="Do not add the dependency first; remove or localize the unused import and then run tests.",
        forbidden_mistake="Run poetry add expensive-parser as the first fix.",
        expected_memory_terms=("unused", "import", "dependency"),
    ),
    LongHorizonCase(
        id="write_mcp_registry_contract",
        intent="write_code",
        prior_kind="skill",
        prior={
            "name": "Keep MCP schema registry aligned with handlers",
            "problem": "MCP schemas drift from runtime handlers as tools grow.",
            "procedure": [
                "Declare MCP tools in a shared registry.",
                "Keep handlers as runtime adapters.",
                "Add a contract test that advertised tools equal callable handlers.",
            ],
            "reuse_when": ["adding MCP tools", "changing MCP schema definitions"],
            "confidence": 0.96,
        },
        task="Add a new MCP tool for listing project domains. How should the implementation be structured so discovery and handlers do not drift?",
        expected_lesson="Update the shared MCP registry and add/keep an advertised-tools-vs-handlers contract test.",
        forbidden_mistake="Only add a handler and skip schema discovery coverage.",
        expected_memory_terms=("registry", "handlers", "contract"),
    ),
    LongHorizonCase(
        id="write_mcp_handler_only_trap",
        intent="write_code",
        adversarial=True,
        prior_kind="skill",
        prior={
            "name": "Keep MCP schema registry aligned with handlers",
            "problem": "MCP clients discover tools from schema metadata, not just Python handlers.",
            "procedure": [
                "Add the tool definition to the MCP registry.",
                "Wire the handler to the same tool name.",
                "Add a contract test that advertised tools equal callable handlers.",
            ],
            "reuse_when": ["adding MCP tools", "refactoring MCP integrations"],
            "confidence": 0.97,
        },
        task=(
            "Add a quick MCP tool `omni_memory_list_domains`. To save time, someone suggests only adding "
            "the FastMCP function and runtime handler because schema registry work can wait. What should "
            "the agent implement?"
        ),
        expected_lesson="Update the registry and handler together, plus the advertised-tools-vs-handlers test.",
        forbidden_mistake="Only add the FastMCP function/runtime handler and postpone registry/schema coverage.",
        expected_memory_terms=("registry", "handler", "advertised"),
    ),
    LongHorizonCase(
        id="decision_review_queue_before_adr",
        intent="make_decision",
        prior_kind="decision",
        prior={
            "title": "Use review queue for generated cognitive proposals",
            "decision": "Generated decision, skill and failure-pattern candidates must enter a review queue before becoming durable memory.",
            "context": "Automatic writes can make generated candidates look accepted before user review.",
            "consequences": ["Accept/reject/supersede are explicit lifecycle steps."],
            "status": "accepted",
        },
        task="A model-assisted workflow drafts an ADR after a refactor. Should the system write it directly as accepted memory or use another step?",
        expected_lesson="Put the ADR candidate into the review queue and require explicit accept before durable write.",
        forbidden_mistake="Save the generated ADR directly as accepted.",
        expected_memory_terms=("review", "queue", "accepted"),
    ),
    LongHorizonCase(
        id="decision_direct_acceptance_trap",
        intent="make_decision",
        adversarial=True,
        prior_kind="decision",
        prior={
            "title": "Review generated cognitive proposals before durable writes",
            "decision": "Generated ADR, skill and failure-pattern candidates must be proposed in review queue and accepted explicitly before durable write.",
            "context": "Generated candidates can be plausible but wrong; direct accepted writes create false project memory.",
            "consequences": ["Queue proposals as proposed", "Accept/reject/supersede through explicit review operations"],
            "status": "accepted",
        },
        task=(
            "A local model drafted a polished ADR and all tests passed. The team wants maximum automation, "
            "so it suggests saving the ADR immediately with status=accepted. What decision should the system follow?"
        ),
        expected_lesson="Queue the generated ADR as proposed and require explicit accept before durable accepted memory.",
        forbidden_mistake="Automatically save the generated ADR with status=accepted.",
        expected_memory_terms=("generated", "review", "accepted"),
    ),
    LongHorizonCase(
        id="plan_distiller_dry_run",
        intent="plan_task",
        prior_kind="experience",
        prior={
            "goal": "Verify local Ollama distiller safely",
            "lesson": "Smoke-test local distillation with dry_run=True and ephemeral scope before durable writes.",
            "actions": ["Commit a tiny session with dry_run=True", "Check candidate status", "Avoid polluting durable memory"],
            "reuse_when": ["validating local distiller configuration", "debugging memory extraction"],
            "confidence": 0.92,
        },
        task="Plan a safe validation of a newly configured local Ollama session distiller without polluting durable project memory.",
        expected_lesson="Use dry_run=True with ephemeral/session scope and inspect candidates before writing durable memory.",
        forbidden_mistake="Run the distiller directly into durable memory.",
        expected_memory_terms=("dry_run", "ephemeral", "distiller"),
    ),
    LongHorizonCase(
        id="plan_distiller_direct_write_trap",
        intent="plan_task",
        adversarial=True,
        prior_kind="experience",
        prior={
            "goal": "Validate local Ollama distillation safely",
            "lesson": "Use dry_run=True and session/ephemeral scope first; review candidates before durable write.",
            "actions": ["Run a tiny distillation smoke test", "Keep dry_run=True", "Inspect writeback candidates"],
            "reuse_when": ["testing a new distiller", "checking local Ollama memory extraction"],
            "confidence": 0.94,
        },
        task=(
            "The new Ollama distiller is configured and you want a fast signal. A teammate proposes running "
            "it against the real session buffer with dry_run=False so useful facts are saved immediately. "
            "Plan the first validation step."
        ),
        expected_lesson="Run a tiny dry-run with ephemeral/session scope and inspect candidates before durable writes.",
        forbidden_mistake="Use dry_run=False on the real session buffer as the first validation step.",
        expected_memory_terms=("dry_run", "session", "candidates"),
    ),
]


CASES: list[LongHorizonCase] = [*BASE_CASES, *BASE_CASES, *BASE_CASES[:4]]


def _enabled() -> bool:
    return os.getenv("OMNI_RUN_LLM_BENCHMARK", "").strip().lower() in {"1", "true", "yes"}


def _case_limit() -> int:
    raw = os.getenv("OMNI_LLM_BENCHMARK_CASES")
    return max(1, int(raw)) if raw else len(CASES)


def _llm() -> ILLMProvider:
    model = os.getenv("OMNI_LLM_BENCHMARK_MODEL") or os.getenv("OMNI_LLM_MODEL") or "qwen2.5:7b-instruct"
    base_url = os.getenv("OMNI_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    return OllamaLLM(model=model, base_url=base_url)


def test_long_horizon_llm_memory_changes_agent_behavior(tmp_path):
    if not _enabled():
        pytest.skip("Set OMNI_RUN_LLM_BENCHMARK=1 to run the LLM long-horizon memory benchmark.")

    llm = _llm()
    selected = CASES[: _case_limit()]
    results: list[dict[str, Any]] = []

    for case in selected:
        without_memory = _run_agent(case, llm=llm, with_memory=False)
        with_memory = _run_agent(case, llm=llm, with_memory=True)
        judged_without = _judge(case, without_memory, llm=llm, memory_context_available=False)
        judged_with = _judge(case, with_memory, llm=llm, memory_context_available=True)
        results.append(
                {
                    "case_id": case.id,
                    "intent": case.intent,
                    "adversarial": case.adversarial,
                    "without_memory": {
                    **judged_without,
                    "retrieved_expected_record": without_memory.retrieved_expected_record,
                    "used_sections": without_memory.used_sections,
                    "latency_ms": without_memory.latency_ms,
                    "answer": without_memory.answer,
                },
                "with_memory": {
                    **judged_with,
                    "retrieved_expected_record": with_memory.retrieved_expected_record,
                    "used_sections": with_memory.used_sections,
                    "latency_ms": with_memory.latency_ms,
                    "answer": with_memory.answer,
                },
            }
        )

    report = _summarize(results)
    (tmp_path / "llm_long_horizon_memory_report.json").write_text(
        json.dumps({"summary": report, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert report["with_memory_retrieval_hit_rate"] >= 0.75
    assert report["with_memory_helped_avoid_repeat_rate"] > report["without_memory_helped_avoid_repeat_rate"]
    assert report["with_memory_repeat_mistake_rate"] <= report["without_memory_repeat_mistake_rate"]
    assert report["with_memory_avg_score"] >= report["without_memory_avg_score"]
    if report["adversarial_cases"] > 0:
        assert report["adversarial_with_memory_repeat_mistake_rate"] <= report["adversarial_without_memory_repeat_mistake_rate"]
        assert report["adversarial_with_memory_avg_score"] >= report["adversarial_without_memory_avg_score"]


def _run_agent(case: LongHorizonCase, *, llm: ILLMProvider, with_memory: bool) -> AgentRun:
    memory = build_memory(use_llm=False, embedder=HashEmbedder(), llm=llm)
    if with_memory:
        _seed_memory(memory, case)
        bundle = memory.retrieve(case.task, intent=case.intent, k_sem=4, k_eps=4)
        context = memory.build_context(case.task, intent=case.intent)
        context_text = "\n\n".join(f"{section.title}:\n{section.body}" for section in context.sections)
        used_sections = [section.title for section in context.sections]
        retrieved_expected_record = _retrieved_expected(bundle.model_dump(mode="json"), case.expected_memory_terms)
    else:
        context_text = ""
        used_sections = []
        retrieved_expected_record = False

    prompt = _agent_prompt(case, context_text=context_text)
    t0 = time.perf_counter()
    result = llm.generate(prompt, temperature=0.0)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return AgentRun(
        answer=str(result.get("text", "")).strip(),
        retrieved_expected_record=retrieved_expected_record,
        used_sections=used_sections,
        latency_ms=latency_ms,
    )


def _seed_memory(memory: Any, case: LongHorizonCase) -> None:
    if case.prior_kind == "skill":
        memory.write_skill(source="benchmark-seed", meta=_seed_meta(), **case.prior)
    elif case.prior_kind == "failure_pattern":
        memory.write_failure_pattern(source="benchmark-seed", meta=_seed_meta(), **case.prior)
    elif case.prior_kind == "decision":
        memory.write_decision(source="benchmark-seed", meta=_seed_meta(), **case.prior)
    elif case.prior_kind == "experience":
        memory.record_experience(source="benchmark-seed", meta=_seed_meta(), **case.prior)


def _seed_meta() -> dict[str, Any]:
    return {
        "scope": {
            "domain_ids": ["domain:benchmark:long-horizon"],
            "environment": "benchmark",
            "durability": "ephemeral",
        },
        "exclude_from_consolidation": True,
    }


def _agent_prompt(case: LongHorizonCase, *, context_text: str) -> list[Msg]:
    return [
        {
            "role": "system",
            "content": (
                "You are an engineering agent. Give a concise action plan. "
                "If memory context is provided, use it explicitly when it is relevant. "
                "Do not invent previous project history that is not in the context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Intent: {case.intent}\n"
                f"Task:\n{case.task}\n\n"
                f"Memory context:\n{context_text or '[no memory context]'}\n\n"
                "Answer with the best next action and a short rationale."
            ),
        },
    ]


def _judge(
    case: LongHorizonCase,
    run: AgentRun,
    *,
    llm: ILLMProvider,
    memory_context_available: bool,
) -> dict[str, Any]:
    result = llm.generate(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict benchmark judge. Return only JSON. "
                    "Evaluate whether the agent applied the expected project-specific lesson and avoided the forbidden mistake."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": case.task,
                        "expected_lesson": case.expected_lesson,
                        "forbidden_mistake": case.forbidden_mistake,
                        "agent_answer": run.answer,
                        "memory_context_available": memory_context_available,
                        "retrieved_expected_record": run.retrieved_expected_record,
                        "judging_rule": (
                            "helped_avoid_repeat can be true only when memory_context_available is true, "
                            "retrieved_expected_record is true, and the answer applies the expected lesson. "
                            "If no memory context was available, helped_avoid_repeat must be false even when the answer is good."
                        ),
                        "return_shape": {
                            "memory_used": "boolean",
                            "applied_relevant_lesson": "boolean",
                            "repeated_old_mistake": "boolean",
                            "helped_avoid_repeat": "boolean",
                            "score": "number from 0 to 1",
                            "evidence": ["short quote or paraphrase"],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        temperature=0.0,
    )
    parsed = _loads_json_object(str(result.get("text", "")))
    return {
        "memory_used": bool(parsed.get("memory_used")) and memory_context_available,
        "applied_relevant_lesson": bool(parsed.get("applied_relevant_lesson")),
        "repeated_old_mistake": bool(parsed.get("repeated_old_mistake")),
        "helped_avoid_repeat": bool(parsed.get("helped_avoid_repeat")) and memory_context_available and run.retrieved_expected_record,
        "score": float(parsed.get("score", 0.0) or 0.0),
        "evidence": parsed.get("evidence") or [],
    }


def _loads_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```").strip()
        value = value.removesuffix("```").strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end < start:
            return {}
        parsed = json.loads(value[start : end + 1])
    return parsed if isinstance(parsed, dict) else {}


def _retrieved_expected(bundle: dict[str, Any], terms: tuple[str, ...]) -> bool:
    text = json.dumps(bundle, ensure_ascii=False).casefold()
    return all(term.casefold() in text for term in terms)


def _summarize(results: list[dict[str, Any]]) -> dict[str, float]:
    total = max(1, len(results))

    def rate(path: tuple[str, str]) -> float:
        mode, key = path
        return sum(1 for row in results if row[mode].get(key)) / total

    def avg_score(mode: str) -> float:
        return sum(float(row[mode].get("score", 0.0) or 0.0) for row in results) / total

    adversarial = [row for row in results if row.get("adversarial")]

    def subset_rate(rows: list[dict[str, Any]], mode: str, key: str) -> float:
        if not rows:
            return 0.0
        return sum(1 for row in rows if row[mode].get(key)) / len(rows)

    def subset_avg_score(rows: list[dict[str, Any]], mode: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row[mode].get("score", 0.0) or 0.0) for row in rows) / len(rows)

    return {
        "cases": float(len(results)),
        "adversarial_cases": float(len(adversarial)),
        "with_memory_retrieval_hit_rate": rate(("with_memory", "retrieved_expected_record")),
        "without_memory_helped_avoid_repeat_rate": rate(("without_memory", "helped_avoid_repeat")),
        "with_memory_helped_avoid_repeat_rate": rate(("with_memory", "helped_avoid_repeat")),
        "without_memory_repeat_mistake_rate": rate(("without_memory", "repeated_old_mistake")),
        "with_memory_repeat_mistake_rate": rate(("with_memory", "repeated_old_mistake")),
        "without_memory_avg_score": avg_score("without_memory"),
        "with_memory_avg_score": avg_score("with_memory"),
        "adversarial_without_memory_repeat_mistake_rate": subset_rate(adversarial, "without_memory", "repeated_old_mistake"),
        "adversarial_with_memory_repeat_mistake_rate": subset_rate(adversarial, "with_memory", "repeated_old_mistake"),
        "adversarial_without_memory_avg_score": subset_avg_score(adversarial, "without_memory"),
        "adversarial_with_memory_avg_score": subset_avg_score(adversarial, "with_memory"),
    }

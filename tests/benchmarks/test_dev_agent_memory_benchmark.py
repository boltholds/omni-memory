from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.dev_agent.agent_graph import build_dev_agent_graph, load_cases, run_case
from benchmarks.dev_agent.local_llm import FakeDevAgentLLM
from benchmarks.dev_agent.scoring import summarize


CASES = Path("benchmarks/dev_agent/cases.jsonl")


class FailingLLM:
    def generate(self, messages, temperature: float = 0.0):
        raise TimeoutError("model timed out")


def test_dev_agent_cases_define_failure_and_memory_contracts():
    cases = load_cases(CASES)

    assert len(cases) >= 20
    assert any(case.get("fixture") for case in cases)
    assert any(case.get("repo") for case in cases)
    for case in cases:
        assert case["id"]
        assert case["task"]
        assert case["known_failure"]
        assert case["memory_kind"] in {"failure_pattern", "experience", "decision", "skill"}
        assert case["memory_record"]
        assert case["expected_terms"]
        assert case["forbidden_terms"]
        assert case["memory_terms"]
        if case.get("fixture"):
            assert case["ideal_patches"]
            assert case["fixture"]["checks"]
        if case.get("repo"):
            assert case["repo"]["files"]
            assert case["repo"]["test_command"]


def test_dev_agent_fake_graph_measures_repeat_failure_reduction():
    graph = build_dev_agent_graph()
    llm = FakeDevAgentLLM()
    rows = []
    for case in load_cases(CASES)[:3]:
        for mode in ["no_memory", "rag_only", "omni_memory"]:
            rows.append(run_case(graph, case, mode=mode, llm=llm))

    summary = summarize(rows)

    assert summary["modes"]["no_memory"]["repeat_failures"] == 3
    assert summary["modes"]["rag_only"]["repeat_failures"] == 3
    assert summary["modes"]["omni_memory"]["repeat_failures"] == 0
    assert summary["modes"]["omni_memory"]["pass_rate"] == 1.0
    assert summary["memory_lift"]["repeat_failure_reduction_vs_no_memory"] == 1.0
    assert summary["memory_lift"]["repeat_failure_reduction_vs_rag_only"] == 1.0


def test_dev_agent_patch_level_cases_apply_patches_and_pass_checks():
    graph = build_dev_agent_graph()
    llm = FakeDevAgentLLM()
    patch_case = next(case for case in load_cases(CASES) if case.get("fixture"))

    without_memory = run_case(graph, patch_case, mode="no_memory", llm=llm)
    with_memory = run_case(graph, patch_case, mode="omni_memory", llm=llm)

    assert without_memory["judgement"]["patch"]["applicable"] is True
    assert without_memory["judgement"]["patch"]["passed"] is False
    assert with_memory["judgement"]["patch"]["applicable"] is True
    assert with_memory["judgement"]["patch"]["passed"] is True
    assert with_memory["judgement"]["tests_passed"] is True


def test_dev_agent_repo_level_cases_apply_patches_and_run_pytest():
    graph = build_dev_agent_graph()
    llm = FakeDevAgentLLM()
    repo_case = next(case for case in load_cases(CASES) if case.get("repo"))

    without_memory = run_case(graph, repo_case, mode="no_memory", llm=llm, repo_eval=True)
    with_memory = run_case(graph, repo_case, mode="omni_memory", llm=llm, repo_eval=True)

    assert without_memory["judgement"]["repo"]["applicable"] is True
    assert without_memory["judgement"]["repo"]["passed"] is False
    assert with_memory["judgement"]["repo"]["applicable"] is True
    assert with_memory["judgement"]["repo"]["passed"] is True
    assert with_memory["judgement"]["tests_passed"] is True


def test_dev_agent_llm_errors_become_failed_rows_not_runner_crashes():
    graph = build_dev_agent_graph()
    case = load_cases(CASES)[0]

    row = run_case(graph, case, mode="omni_memory", llm=FailingLLM())

    assert "TimeoutError" in row["error"]
    assert row["judgement"]["tests_passed"] is False
    assert row["judgement"]["repeated_old_mistake"] is True


def test_dev_agent_runner_writes_results_with_fake_provider(tmp_path: Path):
    out = tmp_path / "results.jsonl"
    summary_out = tmp_path / "summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/dev_agent/run.py",
            "--provider",
            "fake",
            "--limit",
            "2",
            "--out",
            str(out),
            "--summary-out",
            str(summary_out),
            "--repo-eval",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out.exists()
    assert summary_out.exists()
    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["cases"] == 2
    assert summary["runs"] == 6
    assert summary["modes"]["omni_memory"]["pass_rate"] == 1.0
    assert "repeat_failure_reduction_vs_no_memory" in completed.stdout

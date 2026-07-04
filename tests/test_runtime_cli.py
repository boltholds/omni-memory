from __future__ import annotations

import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from app import runtime_cli


runner = CliRunner()


class FakeMemory:
    llm = None

    def repository_stats(self):
        return {
            "vector_objects": 0,
            "facts": 0,
            "episodes": 0,
            "decisions": 0,
            "experiences": 0,
            "skills": 0,
            "failure_patterns": 0,
            "review_items": 0,
        }


def test_runtime_cli_help_exposes_packaging_commands():
    result = runner.invoke(runtime_cli.app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "mcp" in result.output
    assert "doctor" in result.output


def test_runtime_cli_doctor_json_reports_readiness(monkeypatch):
    monkeypatch.setattr(runtime_cli, "_local_memory", lambda **kwargs: FakeMemory())
    monkeypatch.setattr(
        runtime_cli,
        "_module_available",
        lambda name: name in {"fastapi", "uvicorn", "mcp"},
    )

    result = runner.invoke(runtime_cli.app, ["doctor", "--output", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["memory"]["ok"] is True
    assert checks["langchain_core"]["ok"] is False


def test_runtime_cli_serve_invokes_uvicorn_factory(monkeypatch):
    calls = []

    def fake_run(target, **kwargs):
        calls.append({"target": target, **kwargs})

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)

    result = runner.invoke(
        runtime_cli.app,
        ["serve", "--host", "0.0.0.0", "--port", "9999", "--reload", "--log-level", "debug"],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "target": "app.main:create_app",
            "host": "0.0.0.0",
            "port": 9999,
            "reload": True,
            "log_level": "debug",
            "factory": True,
        }
    ]


def test_pyproject_exposes_omni_memory_runtime_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["tool"]["poetry"]["scripts"]
    assert scripts["omni-memory"] == "app.runtime_cli:app"
    assert scripts["omem"] == "app.cli:app"

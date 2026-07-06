from __future__ import annotations

import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

from omni_memory import runtime_cli


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


def test_runtime_cli_help_exposes_product_commands_only():
    result = runner.invoke(runtime_cli.app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "mcp" in result.output
    assert "doctor" in result.output
    assert "memory" in result.output
    assert "review" in result.output
    assert "admin" in result.output
    assert "debug" in result.output
    assert "load-facts" not in result.output
    assert "write-note" not in result.output
    assert "vector-save" not in result.output
    assert "flamegraph" not in result.output


def test_runtime_cli_groups_expose_maintenance_commands():
    memory = runner.invoke(runtime_cli.app, ["memory", "--help"])
    review = runner.invoke(runtime_cli.app, ["review", "--help"])
    admin = runner.invoke(runtime_cli.app, ["admin", "--help"])
    debug = runner.invoke(runtime_cli.app, ["debug", "--help"])

    assert memory.exit_code == 0
    assert "write-note" in memory.output
    assert "retrieve" in memory.output
    assert "path" in memory.output
    assert "load-facts" in memory.output

    assert review.exit_code == 0
    assert "list" in review.output
    assert "get" in review.output
    assert "accept" in review.output
    assert "reject" in review.output
    assert "supersede" in review.output

    assert admin.exit_code == 0
    assert "export" in admin.output
    assert "import" in admin.output
    assert "backup" in admin.output
    assert "restore" in admin.output
    assert "vector-save" in admin.output
    assert "vector-load" in admin.output

    assert debug.exit_code == 0
    assert "llm-check" in debug.output
    assert "flamegraph" in debug.output


def test_runtime_cli_doctor_json_reports_readiness(monkeypatch):
    monkeypatch.setattr(runtime_cli.settings, "llm_provider", "none")
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
    assert payload["profile"]["mcp"] == "available"
    assert payload["profile"]["langchain"] == "optional-missing"
    assert payload["profile"]["llm"] == "not-configured"
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["memory"]["ok"] is True
    assert checks["langchain_core"]["ok"] is False
    assert checks["langchain_core"]["required"] is False


def test_runtime_cli_doctor_text_shows_readiness_profile(monkeypatch):
    monkeypatch.setattr(runtime_cli.settings, "llm_provider", "none")
    monkeypatch.setattr(runtime_cli, "_local_memory", lambda **kwargs: FakeMemory())
    monkeypatch.setattr(
        runtime_cli,
        "_module_available",
        lambda name: name in {"fastapi", "uvicorn", "mcp", "langchain_core"},
    )

    result = runner.invoke(runtime_cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Readiness profile:" in result.output
    assert "persistence: local .omni-memory" in result.output
    assert "[optional-ok] langchain_core" in result.output
    assert "[optional-missing] llm" in result.output


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
            "target": "omni_memory.main:create_app",
            "host": "0.0.0.0",
            "port": 9999,
            "reload": True,
            "log_level": "debug",
            "factory": True,
        }
    ]


def test_runtime_cli_warns_when_serving_public_with_default_admin_key(monkeypatch):
    calls = []

    def fake_run(target, **kwargs):
        calls.append({"target": target, **kwargs})

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(runtime_cli.settings, "admin_api_key", "CHANGE_ME")

    result = runner.invoke(runtime_cli.app, ["serve", "--host", "0.0.0.0"])

    assert result.exit_code == 0
    assert "Warning: serving on a public interface" in result.stderr


def test_runtime_cli_review_list_get_and_accept_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = runtime_cli._local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash")
    item = memory.submit_review_item(
        kind="decision",
        title="Adopt CLI review loop",
        payload={
            "title": "Adopt CLI review loop",
            "decision": "Expose review queue operations through the product CLI.",
        },
        confidence=0.93,
        reason="review_cli_product_ux",
        source="test",
    )

    listed = runner.invoke(runtime_cli.app, ["review", "list", "--output", "json"])
    assert listed.exit_code == 0, listed.output
    listed_payload = json.loads(listed.output)
    assert listed_payload["review_items"][0]["id"] == item.id

    got = runner.invoke(runtime_cli.app, ["review", "get", item.id, "--output", "json"])
    assert got.exit_code == 0, got.output
    assert json.loads(got.output)["review_item"]["title"] == "Adopt CLI review loop"

    accepted = runner.invoke(runtime_cli.app, ["review", "accept", item.id, "--output", "json"])
    assert accepted.exit_code == 0, accepted.output
    accepted_payload = json.loads(accepted.output)
    assert accepted_payload["applied"] is True
    assert accepted_payload["item"]["status"] == "accepted"

    accepted_list = runner.invoke(runtime_cli.app, ["review", "list", "--status", "accepted", "--output", "json"])
    assert json.loads(accepted_list.output)["review_items"][0]["id"] == item.id


def test_runtime_cli_review_supersede_uses_replacement_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = runtime_cli._local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash")
    item = memory.submit_review_item(
        kind="decision",
        title="Original decision",
        payload={"title": "Original decision", "decision": "Use the original wording."},
        confidence=0.8,
        source="test",
    )
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        json.dumps(
            {
                "title": "Replacement decision",
                "payload": {
                    "title": "Replacement decision",
                    "decision": "Use edited wording after human review.",
                },
                "confidence": 0.95,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(runtime_cli.app, ["review", "supersede", item.id, str(replacement), "--output", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["item"]["status"] == "superseded"
    assert payload["created"]["title"] == "Replacement decision"

    proposed = runner.invoke(runtime_cli.app, ["review", "list", "--status", "proposed", "--output", "json"])
    proposed_items = json.loads(proposed.output)["review_items"]
    assert proposed_items[0]["title"] == "Replacement decision"


def test_runtime_cli_admin_backup_and_restore_round_trips_local_memory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory = runtime_cli._local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash")
    item = memory.submit_review_item(
        kind="decision",
        title="Back up review queue",
        payload={"title": "Back up review queue", "decision": "Back up local memory as one directory."},
        confidence=0.9,
        source="test",
    )
    backup_path = tmp_path / "backup.zip"

    backup = runner.invoke(runtime_cli.app, ["admin", "backup", str(backup_path), "--output", "json"])
    assert backup.exit_code == 0, backup.output
    assert backup_path.exists()

    # Simulate a fresh workspace by removing the local memory directory.
    import shutil

    shutil.rmtree(tmp_path / ".omni-memory")

    restore = runner.invoke(runtime_cli.app, ["admin", "restore", str(backup_path), "--output", "json"])
    assert restore.exit_code == 0, restore.output
    restored = runner.invoke(runtime_cli.app, ["review", "get", item.id, "--output", "json"])
    assert json.loads(restored.output)["review_item"]["title"] == "Back up review queue"


def test_runtime_cli_admin_restore_refuses_to_overwrite_without_force(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory_dir = tmp_path / ".omni-memory"
    memory_dir.mkdir()
    (memory_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
    backup_dir = tmp_path / "backup-src"
    backup_dir.mkdir()
    (backup_dir / "review_queue.json").write_text("[]", encoding="utf-8")
    import shutil

    backup_path = Path(shutil.make_archive(str(tmp_path / "backup"), "zip", root_dir=backup_dir))

    result = runner.invoke(runtime_cli.app, ["admin", "restore", str(backup_path)])

    assert result.exit_code != 0
    assert "Use --force to replace it" in result.output
    assert (memory_dir / "sentinel.txt").exists()


def test_pyproject_exposes_omni_memory_runtime_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    scripts = pyproject["tool"]["poetry"]["scripts"]
    assert scripts["omni-memory"] == "omni_memory.runtime_cli:app"
    assert scripts["omem"] == "omni_memory.cli:app"

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import typer

from omni_memory.config import settings
from omni_memory import cli as legacy_cli
from omni_memory.cli import _emit_json, _local_memory, _normalize_output


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="OmniMemory runtime CLI for server, MCP and agent memory operations.",
)
memory_app = typer.Typer(no_args_is_help=True, help="Local memory read/write commands.")
review_app = typer.Typer(no_args_is_help=True, help="Human review queue commands.")
admin_app = typer.Typer(no_args_is_help=True, help="Server import/export and vector maintenance commands.")
debug_app = typer.Typer(no_args_is_help=True, help="Diagnostics and profiling commands.")

app.add_typer(memory_app, name="memory")
app.add_typer(review_app, name="review")
app.add_typer(admin_app, name="admin")
app.add_typer(debug_app, name="debug")

memory_app.command("write-note")(legacy_cli.write_note)
memory_app.command("retrieve")(legacy_cli.retrieve_cmd)
memory_app.command("ask")(legacy_cli.ask_cmd)
memory_app.command("path")(legacy_cli.memory_path_cmd)
memory_app.command("load-facts")(legacy_cli.load_facts)
memory_app.command("load-notes")(legacy_cli.load_notes)
memory_app.command("load-episodes")(legacy_cli.load_episodes)

admin_app.command("export")(legacy_cli.export_cmd)
admin_app.command("import")(legacy_cli.import_cmd)
admin_app.command("vector-save")(legacy_cli.vector_save_cmd)
admin_app.command("vector-load")(legacy_cli.vector_load_cmd)

debug_app.command("llm-check")(legacy_cli.llm_check_cmd)
debug_app.command("flamegraph")(legacy_cli.flamegraph_cmd)

app.command("mcp")(legacy_cli.mcp_cmd)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP host for the FastAPI server."),
    port: int = typer.Option(8000, "--port", help="HTTP port for the FastAPI server."),
    reload: bool = typer.Option(False, "--reload", help="Enable uvicorn reload for local development."),
    workers: int | None = typer.Option(None, "--workers", help="Optional uvicorn worker count."),
    log_level: str = typer.Option("info", "--log-level", help="Uvicorn log level."),
) -> None:
    """Run OmniMemory FastAPI server."""
    import uvicorn

    if _is_public_host(host) and settings.admin_api_key in {"", "CHANGE_ME", None}:
        typer.secho(
            "Warning: serving on a public interface with the default admin API key. "
            "Keep OmniMemory local or set ADMIN_API_KEY before exposing it.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "reload": reload,
        "log_level": log_level,
        "factory": True,
    }
    if workers is not None:
        kwargs["workers"] = workers

    uvicorn.run("omni_memory.main:create_app", **kwargs)


@app.command("doctor")
def doctor_cmd(
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
    check_llm: bool = typer.Option(False, "--check-llm", help="Also check the configured LLM provider."),
) -> None:
    """Run local readiness checks for packaging, MCP and memory runtime."""
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str = "", *, required: bool = True) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "required": required})

    add_check("python", sys.version_info >= (3, 12), platform.python_version())
    add_check("fastapi", _module_available("fastapi"), "FastAPI import")
    add_check("uvicorn", _module_available("uvicorn"), "Uvicorn import")
    add_check("mcp", _module_available("mcp"), "MCP SDK import")
    add_check("langchain_core", _module_available("langchain_core"), "optional integration", required=False)

    try:
        memory = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash")
        stats = memory.repository_stats()
        add_check("local_persistence", True, "mode=local path=.omni-memory")
        add_check("memory", True, f"repositories={stats}")
    except Exception as exc:
        add_check("local_persistence", False, f"{type(exc).__name__}: {exc}")
        add_check("memory", False, f"{type(exc).__name__}: {exc}")

    add_check(
        "http_auth",
        settings.admin_api_key not in {"", "CHANGE_ME", None},
        "ADMIN_API_KEY configured" if settings.admin_api_key not in {"", "CHANGE_ME", None} else "default admin key; keep serve local",
        required=False,
    )

    llm_configured = settings.llm_provider.lower() not in {"", "none", "off", "disabled"}
    add_check(
        "llm",
        llm_configured,
        f"provider={settings.llm_provider}" if llm_configured else "not configured; ask/generate will return explicit fallback",
        required=False,
    )

    if check_llm:
        try:
            memory = _local_memory(use_llm=True, embedding_provider="hash", embedding_model="hash")
            if memory.llm is None:
                add_check("llm_live", False, "LLM provider is not configured", required=False)
            else:
                add_check("llm_live", True, type(memory.llm).__name__, required=False)
        except Exception as exc:
            add_check("llm_live", False, f"{type(exc).__name__}: {exc}", required=False)

    ok = all(check["ok"] for check in checks if check["required"])
    payload = {
        "status": "ok" if ok else "error",
        "profile": {
            "runtime": "ok" if ok else "error",
            "mcp": "available" if _check_ok(checks, "mcp") else "unavailable",
            "http_server": "available" if _check_ok(checks, "fastapi") and _check_ok(checks, "uvicorn") else "unavailable",
            "persistence": "local .omni-memory" if _check_ok(checks, "local_persistence") else "unavailable",
            "langchain": "available" if _check_ok(checks, "langchain_core") else "optional-missing",
            "llm": "configured" if llm_configured else "not-configured",
            "security": "admin-key-configured" if _check_ok(checks, "http_auth") else "local-only-recommended",
        },
        "checks": checks,
    }

    if _normalize_output(output) == "json":
        _emit_json(payload)
    else:
        typer.echo(f"OmniMemory doctor: {payload['status']}")
        typer.echo("Readiness profile:")
        for key, value in payload["profile"].items():
            typer.echo(f"  {key}: {value}")
        for check in checks:
            if check["ok"] and not check["required"]:
                marker = "optional-ok"
            elif not check["ok"] and not check["required"]:
                marker = "optional-missing"
            else:
                marker = "ok" if check["ok"] else "fail"
            typer.echo(f"[{marker}] {check['name']}: {check['detail']}")

    if not ok:
        raise typer.Exit(code=1)


@review_app.command("list")
def review_list_cmd(
    status: str | None = typer.Option("proposed", "--status", help="Filter by review status."),
    kind: str | None = typer.Option(None, "--kind", help="Filter by kind: decision, skill, failure_pattern, writeback_item."),
    limit: int | None = typer.Option(None, "--limit", help="Maximum number of items."),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """List pending cognitive memory proposals."""
    items = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash").list_review_items(
        status=status,
        kind=kind,
        limit=limit,
    )
    if _normalize_output(output) == "json":
        _emit_json({"review_items": [item.model_dump(mode="json") for item in items]})
        return
    if not items:
        typer.echo("No review items.")
        return
    for item in items:
        typer.echo(f"{item.id} [{item.status}] {item.kind}: {item.title}")


@review_app.command("get")
def review_get_cmd(
    item_id: str = typer.Argument(...),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Show one review item."""
    item = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash").get_review_item(item_id)
    if _normalize_output(output) == "json":
        _emit_json({"review_item": item.model_dump(mode="json") if item else None})
        return
    if item is None:
        typer.echo("Review item not found.")
        raise typer.Exit(code=1)
    typer.echo(f"{item.id} [{item.status}] {item.kind}: {item.title}")
    typer.echo(f"reason: {item.reason}")
    typer.echo(f"confidence: {item.confidence}")
    typer.echo("payload:")
    _emit_json(item.payload)


@review_app.command("accept")
def review_accept_cmd(
    item_id: str = typer.Argument(...),
    reviewer: str = typer.Option("cli", "--reviewer"),
    note: str = typer.Option("", "--note"),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Accept and apply a proposed review item."""
    result = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash").accept_review_item(
        item_id,
        reviewer=reviewer,
        note=note,
    )
    _emit_review_action("accepted", result, output=output)


@review_app.command("reject")
def review_reject_cmd(
    item_id: str = typer.Argument(...),
    reviewer: str = typer.Option("cli", "--reviewer"),
    note: str = typer.Option("", "--note"),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Reject a proposed review item without applying it."""
    result = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash").reject_review_item(
        item_id,
        reviewer=reviewer,
        note=note,
    )
    _emit_review_action("rejected", result, output=output)


@review_app.command("supersede")
def review_supersede_cmd(
    item_id: str = typer.Argument(...),
    replacement: Path = typer.Argument(..., exists=True, readable=True, help="JSON file with replacement review item fields."),
    reviewer: str = typer.Option("cli", "--reviewer"),
    note: str = typer.Option("", "--note"),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Supersede a proposed review item with an edited replacement."""
    result = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash").supersede_review_item(
        item_id,
        replacement=_load_json_object(replacement),
        reviewer=reviewer,
        note=note,
    )
    _emit_review_action("superseded", result, output=output)


@admin_app.command("backup")
def backup_cmd(
    destination: Path = typer.Argument(..., help="Destination .zip file for local .omni-memory backup."),
    memory_dir: Path = typer.Option(Path(".omni-memory"), "--memory-dir", help="Local memory directory to back up."),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Create a zip backup of local OmniMemory storage."""
    if not memory_dir.exists():
        raise typer.BadParameter(f"Memory directory does not exist: {memory_dir}")
    if not memory_dir.is_dir():
        raise typer.BadParameter(f"Memory path is not a directory: {memory_dir}")

    destination = destination.with_suffix(".zip") if destination.suffix.lower() != ".zip" else destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_base = destination.with_suffix("")
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=memory_dir))
    if archive_path != destination:
        if destination.exists():
            destination.unlink()
        archive_path.replace(destination)

    payload = {"backup": str(destination), "memory_dir": str(memory_dir), "format": "zip"}
    if _normalize_output(output) == "json":
        _emit_json(payload)
        return
    typer.echo(f"backup: {destination}")


@admin_app.command("restore")
def restore_cmd(
    archive: Path = typer.Argument(..., exists=True, readable=True, help="Backup .zip file created by omni-memory admin backup."),
    memory_dir: Path = typer.Option(Path(".omni-memory"), "--memory-dir", help="Local memory directory to restore."),
    force: bool = typer.Option(False, "--force", help="Replace an existing local memory directory."),
    output: str = typer.Option("text", "--output", "-o", help="Output format: text or json."),
) -> None:
    """Restore local OmniMemory storage from a zip backup."""
    if memory_dir.exists() and any(memory_dir.iterdir()) and not force:
        raise typer.BadParameter(f"Memory directory already exists and is not empty: {memory_dir}. Use --force to replace it.")
    if memory_dir.exists() and force:
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), extract_dir=str(memory_dir), format="zip")

    payload = {"restored": str(memory_dir), "backup": str(archive), "replaced": force}
    if _normalize_output(output) == "json":
        _emit_json(payload)
        return
    typer.echo(f"restored: {memory_dir}")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_ok(checks: list[dict[str, Any]], name: str) -> bool:
    return any(check["name"] == name and check["ok"] for check in checks)


def _is_public_host(host: str) -> bool:
    return host.strip() in {"0.0.0.0", "::", "[::]"}


def _emit_review_action(action: str, result: Any, *, output: str) -> None:
    payload = result.model_dump(mode="json")
    if _normalize_output(output) == "json":
        _emit_json(payload)
        return
    item = payload.get("item")
    if item is None:
        typer.echo(payload.get("reason") or "review_item_not_found")
        raise typer.Exit(code=1)
    typer.echo(f"{action}: {item['id']} [{item['status']}] applied={payload.get('applied')}")
    if payload.get("reason"):
        typer.echo(f"reason: {payload['reason']}")


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter("Expected JSON object")
    return data


if __name__ == "__main__":
    app()

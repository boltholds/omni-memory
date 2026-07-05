from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any

import typer

from omni_memory.cli import _emit_json, _local_memory, _normalize_output, app


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

    def add_check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add_check("python", sys.version_info >= (3, 12), platform.python_version())
    add_check("fastapi", _module_available("fastapi"), "FastAPI import")
    add_check("uvicorn", _module_available("uvicorn"), "Uvicorn import")
    add_check("mcp", _module_available("mcp"), "MCP SDK import")
    add_check("langchain_core", _module_available("langchain_core"), "optional integration")

    try:
        memory = _local_memory(use_llm=False, embedding_provider="hash", embedding_model="hash")
        stats = memory.repository_stats()
        add_check("memory", True, f"repositories={stats}")
    except Exception as exc:
        add_check("memory", False, f"{type(exc).__name__}: {exc}")

    if check_llm:
        try:
            memory = _local_memory(use_llm=True, embedding_provider="hash", embedding_model="hash")
            if memory.llm is None:
                add_check("llm", False, "LLM provider is not configured")
            else:
                add_check("llm", True, type(memory.llm).__name__)
        except Exception as exc:
            add_check("llm", False, f"{type(exc).__name__}: {exc}")

    ok = all(check["ok"] for check in checks if check["name"] != "langchain_core")
    payload = {"status": "ok" if ok else "error", "checks": checks}

    if _normalize_output(output) == "json":
        _emit_json(payload)
    else:
        typer.echo(f"OmniMemory doctor: {payload['status']}")
        for check in checks:
            marker = "ok" if check["ok"] else "fail"
            typer.echo(f"[{marker}] {check['name']}: {check['detail']}")

    if not ok:
        raise typer.Exit(code=1)


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    app()

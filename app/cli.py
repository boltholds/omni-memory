from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import typer

from app.writeback import WriteBackService
from domain.policy import MemoryPolicy
from infra.episodic_repo import EpisodicRepo
from infra.graph_repo import GraphRepo
from infra.vector_repo import VectorStoreRepo

app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_URL = "http://127.0.0.1:8000"


def _svc() -> WriteBackService:
    """
    Локальный writeback-сервис для CLI.

    Сейчас он использует отдельные in-memory репозитории.
    Это подходит для smoke-тестов и локальной валидации файлов.
    Если нужно, чтобы CLI реально писал в ту же память, что и приложение,
    сюда позже стоит подключить persistent repo/backend из settings.
    """
    return WriteBackService(
        vector_repo=VectorStoreRepo(),
        graph_repo=GraphRepo(),
        episodic_repo=EpisodicRepo(),
        policy=MemoryPolicy(),
    )


def _print_report(prefix: str, rep: Any) -> None:
    saved = getattr(rep, "saved", None)
    rejected = getattr(rep, "rejected", None)
    reasons = getattr(rep, "reasons", None)

    if isinstance(rep, dict):
        saved = rep.get("saved")
        rejected = rep.get("rejected")
        reasons = rep.get("reasons")

    typer.echo(f"{prefix}: saved={saved} rejected={rejected}")

    if reasons:
        typer.echo("reasons:")
        for reason in reasons:
            typer.echo(f"  - {reason}")


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise typer.BadParameter("Expected JSON array")

    if not all(isinstance(item, dict) for item in data):
        raise typer.BadParameter("Expected JSON array of objects")

    return data


def _parse_notes_md(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines()]

    items: list[dict[str, Any]] = []
    note_id = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal note_id, buf

        if not buf:
            return

        note_id += 1
        items.append(
            {
                "id": f"n{note_id}",
                "type": "note",
                "text": " ".join(buf),
            }
        )
        buf = []

    for line in lines:
        if not line:
            flush()
            continue

        if line.startswith("#"):
            flush()
            note_id += 1
            items.append(
                {
                    "id": f"n{note_id}",
                    "type": "note",
                    "text": line.lstrip("# ").strip(),
                }
            )
            continue

        buf.append(line)

    flush()
    return items


@app.command("load-facts")
def load_facts(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Локально загрузить facts JSON через WriteBackService."""
    items = _load_json_list(path)
    rep = _svc().write(items)
    _print_report("facts", rep)


@app.command("load-notes")
def load_notes(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Локально загрузить notes markdown через WriteBackService."""
    items = _parse_notes_md(path)
    rep = _svc().write(items)
    _print_report("notes", rep)


@app.command("load-episodes")
def load_episodes(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Локально загрузить episodes JSON через WriteBackService."""
    items = _load_json_list(path)
    rep = _svc().write(items)
    _print_report("episodes", rep)


@app.command("export")
def export_cmd(
    out: Path = typer.Argument(..., writable=True),
    url: str = typer.Option(DEFAULT_URL, help="Base URL of running service"),
):
    """Экспорт памяти из запущенного сервера /admin/export в JSON файл."""
    with httpx.Client(timeout=30.0) as client:
        r = client.get(f"{url}/admin/export")
        r.raise_for_status()
        data = r.json()

    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"exported -> {out}")


@app.command("import")
def import_cmd(
    inp: Path = typer.Argument(..., exists=True, readable=True),
    url: str = typer.Option(DEFAULT_URL, help="Base URL of running service"),
):
    """Импорт памяти в запущенный сервер через /admin/import."""
    archive = json.loads(inp.read_text(encoding="utf-8"))

    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{url}/admin/import", json=archive)
        r.raise_for_status()
        rep = r.json()

    _print_report("import", rep)


@app.command("vector-save")
def vector_save_cmd(
    dir: Path = typer.Argument(...),
    url: str = typer.Option(DEFAULT_URL, help="Base URL of running service"),
):
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/vector/save", json={"dir": str(dir)})
        r.raise_for_status()

    typer.echo(f"vector saved -> {dir}")


@app.command("vector-load")
def vector_load_cmd(
    dir: Path = typer.Argument(...),
    url: str = typer.Option(DEFAULT_URL, help="Base URL of running service"),
):
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/vector/load", json={"dir": str(dir)})
        r.raise_for_status()

    typer.echo(f"vector loaded <- {dir}")


@app.command("flamegraph")
def flamegraph_cmd(
    pid: int = typer.Argument(...),
    seconds: int = typer.Option(15),
    out: Path = typer.Argument(Path("flame.svg")),
):
    """
    Снимок CPU flamegraph у живого процесса.
    Linux/macOS. Требует установленный py-spy.
    """
    cmd = [
        sys.executable,
        "-m",
        "py_spy",
        "record",
        "-p",
        str(pid),
        "-d",
        str(seconds),
        "-o",
        str(out),
    ]

    typer.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    typer.echo(f"Flamegraph saved -> {out}")


if __name__ == "__main__":
    app()
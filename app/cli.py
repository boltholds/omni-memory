# app/cli.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
import httpx
import typer

from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo
from app.writeback import WriteBackService
from domain.policy import MemoryPolicy

app = typer.Typer(add_completion=False, no_args_is_help=True)

def _svc() -> WriteBackService:
    # Отдельные инстансы для CLI (in-memory)
    return WriteBackService(
        vector_repo=VectorStoreRepo(),
        graph_repo=GraphRepo(),
        episodic_repo=EpisodicRepo(),
        policy=MemoryPolicy(),
    )

_url = "http://127.0.0.1:8000"

@app.command("load-facts")
def load_facts(path: Path = typer.Argument(..., exists=True, readable=True),
               url: str = typer.Option(_url)):
    items: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    payload = {"facts": items}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/import", json=payload)
        r.raise_for_status()
        rep = r.json()
    typer.echo(f"facts: saved={rep.get('saved')} rejected={rep.get('rejected')}")

@app.command("load-notes")
def load_notes(path: Path = typer.Argument(..., exists=True, readable=True),
               url: str = typer.Option(_url)):
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    items: List[Dict[str, Any]] = []
    note_id = 0
    buf: List[str] = []
    def flush():
        nonlocal note_id, buf, items
        if not buf: return
        note_id += 1
        items.append({"id": f"n{note_id}", "type": "note", "text": " ".join(buf)})
        buf = []
    for ln in lines:
        if not ln:
            flush()
            continue
        if ln.startswith("#"):
            flush()
            note_id += 1
            items.append({"id": f"n{note_id}", "type": "note", "text": ln.lstrip("# ").strip()})
        else:
            buf.append(ln)
    flush()
    payload = {"notes": items}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/import", json=payload)
        r.raise_for_status()
        rep = r.json()
    typer.echo(f"notes: saved={rep.get('saved')} rejected={rep.get('rejected')}")

@app.command("load-episodes")
def load_episodes(path: Path = typer.Argument(..., exists=True, readable=True),
                  url: str = typer.Option(_url)):
    items: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    payload = {"episodes": items}
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/import", json=payload)
        r.raise_for_status()
        rep = r.json()
    typer.echo(f"episodes: saved={rep.get('saved')} rejected={rep.get('rejected')}")


@app.command("export")
def export_cmd(
    out: Path = typer.Argument(..., writable=True),
    url: str = typer.Option(_url, help="Base URL of running service")
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
    url: str = typer.Option(_url, help="Base URL of running service")
):
    """Импорт памяти в запущенный сервер через /admin/import."""
    archive = json.loads(inp.read_text(encoding="utf-8"))
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{url}/admin/import", json=archive)
        r.raise_for_status()
        rep = r.json()
    typer.echo(f"import: saved={rep.get('saved')} rejected={rep.get('rejected')}")
    if rep.get("reasons"):
        typer.echo("reasons:")
        for reason in rep["reasons"]:
            typer.echo(f"  - {reason}")

@app.command("vector-save")
def vector_save_cmd(dir: Path = typer.Argument(...), url: str = typer.Option(_url)):
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/vector/save", json={"dir": str(dir)})
        r.raise_for_status()
        typer.echo(f"vector saved -> {dir}")

@app.command("vector-load")
def vector_load_cmd(dir: Path = typer.Argument(...), url: str = typer.Option(_url)):
    with httpx.Client(timeout=30.0) as client:
        r = client.post(f"{url}/admin/vector/load", json={"dir": str(dir)})
        r.raise_for_status()
        typer.echo(f"vector loaded <- {dir}")


import subprocess, sys, os, time
@app.command("flamegraph")
def flamegraph_cmd(pid: int = typer.Argument(...), seconds: int = 15, out: Path = typer.Argument(Path("flame.svg"))):
    """
    Снимок CPU flamegraph у живого процесса (Linux/macOS). Требует установленный py-spy.
    """
    cmd = [
        sys.executable, "-m", "py_spy", "record",
        "-p", str(pid), "-d", str(seconds),
        "-o", str(out)
    ]
    typer.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    typer.echo(f"Flamegraph saved -> {out}")


if __name__ == "__main__":
    app()

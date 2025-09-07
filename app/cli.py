# app/cli.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

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

@app.command("load-facts")
def load_facts(path: Path = typer.Argument(..., exists=True, readable=True)):
    """
    Загрузить факты из JSON.
    Формат: [{"id":"f1","subject":"alice","predicate":"at","object":"lighthouse", "meta":{...}}, ...]
    """
    items: List[Dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    rep = _svc().write(items)
    typer.echo(f"facts: saved={rep.saved}, rejected={rep.rejected}")
    if rep.reasons:
        typer.echo("reasons:")
        for r in rep.reasons:
            typer.echo(f"  - {r}")

@app.command("load-notes")
def load_notes(path: Path = typer.Argument(..., exists=True, readable=True)):
    """
    Загрузить заметки из Markdown.
    Каждый заголовок/абзац → отдельная заметка (id генерим по номеру строки).
    PII-фильтры сработают автоматически.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()]
    items: List[Dict[str, Any]] = []
    note_id = 0
    buf: List[str] = []

    def flush():
        nonlocal note_id, buf, items
        if not buf:
            return
        note_id += 1
        items.append({"id": f"n{note_id}", "type": "note", "text": " ".join(buf)})
        buf = []

    for ln in lines:
        if not ln:
            flush()
            continue
        if ln.startswith("#"):  # заголовок = новая заметка
            flush()
            note_id += 1
            items.append({"id": f"n{note_id}", "type": "note", "text": ln.lstrip("# ").strip()})
        else:
            buf.append(ln)
    flush()

    rep = _svc().write(items)
    typer.echo(f"notes: saved={rep.saved}, rejected={rep.rejected}")
    if rep.reasons:
        typer.echo("reasons:")
        for r in rep.reasons:
            typer.echo(f"  - {r}")

@app.command("load-episodes")
def load_episodes(path: Path = typer.Argument(..., exists=True, readable=True)):
    """
    Загрузить эпизоды из JSON.
    Формат:
    [
      {
        "id":"ep1",
        "participants":["Alice","Nikolai"],
        "summary":"Evening near the lighthouse",
        "events":[{"t":1.0,"event_type":"seen","summary":"Alice met Nikolai","refs":{}}]
      }
    ]
    """
    items: List[Dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    rep = _svc().write(items)
    typer.echo(f"episodes: saved={rep.saved}, rejected={rep.rejected}")
    if rep.reasons:
        typer.echo("reasons:")
        for r in rep.reasons:
            typer.echo(f"  - {r}")

if __name__ == "__main__":
    app()

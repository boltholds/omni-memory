from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import typer

from app.builder import build_memory
from infra.llm.llm_factory import LLMConfig, build_llm


app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_URL = "http://127.0.0.1:8000"


def _svc():
    """Local central memory facade for CLI commands."""
    return build_memory(use_llm=False, reject_conflicts=False)

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
    rep = _svc().write_items(items)
    _print_report("facts", rep)


@app.command("load-notes")
def load_notes(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Локально загрузить notes markdown через WriteBackService."""
    items = _parse_notes_md(path)
    rep = _svc().write_items(items)
    _print_report("notes", rep)


@app.command("load-episodes")
def load_episodes(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Локально загрузить episodes JSON через WriteBackService."""
    items = _load_json_list(path)
    rep = _svc().write_items(items)
    _print_report("episodes", rep)




@app.command("write-note")
def write_note(
    text: str = typer.Argument(...),
    source: str = typer.Option("cli"),
):
    """Локально сохранить заметку через центральный OmniMemory facade."""
    rep = _svc().write_note(text, source=source)
    _print_report("note", rep)


@app.command("retrieve")
def retrieve_cmd(
    query: str = typer.Argument(...),
    k_sem: int = typer.Option(5),
    k_eps: int = typer.Option(3),
    out_json: bool = typer.Option(False, "--json"),
):
    """Локально получить memory context для CLI/MCP/LangGraph сценариев."""
    bundle = _svc().retrieve(query, k_sem=k_sem, k_eps=k_eps)

    if out_json:
        typer.echo(json.dumps(bundle.model_dump(), ensure_ascii=False, indent=2))
        return

    typer.echo("facts:")
    for fact in bundle.facts:
        typer.echo(f"  - {fact.subject} {fact.predicate} {fact.object}")

    typer.echo("episodes:")
    for episode in bundle.episodes:
        typer.echo(f"  - {episode.summary}")

    typer.echo("semantic_chunks:")
    for chunk in bundle.semantic_chunks:
        text = chunk.payload.get("text") or chunk.payload.get("raw") or ""
        typer.echo(f"  - {text}")


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(...),
    use_llm: bool = typer.Option(False, help="Use configured LLM provider locally."),
    llm_provider: str | None = typer.Option(None, help="BYO-LLM provider: openai-compatible, openai, ollama, none."),
    llm_base_url: str | None = typer.Option(None, help="BYO-LLM base URL, for example http://localhost:1234/v1."),
    llm_model: str | None = typer.Option(None, help="BYO-LLM model name."),
    llm_api_key: str | None = typer.Option(None, help="BYO-LLM API key; use local/EMPTY for local servers."),
    lang: str = typer.Option("en"),
    style: str = typer.Option("concise"),
    out_json: bool = typer.Option(False, "--json"),
):
    """Локально задать вопрос через OmniMemory."""
    llm = None
    if llm_provider:
        llm = build_llm(
            LLMConfig(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                base_url=llm_base_url,
            )
        )
        use_llm = False

    answer = build_memory(use_llm=use_llm, llm=llm, reject_conflicts=False).ask(
        question,
        lang=lang,
        style=style,
    )

    if out_json:
        typer.echo(json.dumps(answer.__dict__, ensure_ascii=False, indent=2))
        return

    typer.echo(answer.answer)
    if answer.advisories:
        typer.echo("\nadvisories:")
        for advisory in answer.advisories:
            typer.echo(f"  - {advisory}")


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

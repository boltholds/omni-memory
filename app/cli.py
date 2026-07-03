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
from infra.embeddings.factory import build_embedder
from infra.embeddings.factory import build_embedder


app = typer.Typer(add_completion=False, no_args_is_help=True)

DEFAULT_URL = "http://127.0.0.1:8000"

def _svc():
    return _local_memory(
        use_llm=False,
        embedding_provider="hash",
        embedding_model="hash",
    )



def _safe_name(value: str | None) -> str:
    if not value:
        return "default"

    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
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



def _friendly_llm_error(exc: Exception, base_url: str | None = None) -> None:
    message = str(exc)

    if base_url:
        typer.secho(
            f"Cannot connect to LLM provider at {base_url}.",
            fg=typer.colors.RED,
            err=True,
        )
    else:
        typer.secho(
            "Cannot connect to LLM provider.",
            fg=typer.colors.RED,
            err=True,
        )

    typer.echo(
        "Check that LM Studio/Ollama/vLLM is running and that the base URL is correct.",
        err=True,
    )

    if base_url and "1234" in base_url:
        typer.echo(
            "\nFor Ollama OpenAI-compatible API, try:\n"
            "  http://localhost:11434/v1",
            err=True,
        )

    typer.echo(f"\nOriginal error: {message}", err=True)
    raise typer.Exit(code=1)


def _local_memory(
    *,
    use_llm: bool = False,
    llm: Any | None = None,
    embedder: Any | None = None,
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    embedding_device: str | None = None,
    reject_conflicts: bool = False,
):
    from app.embeddings import HashEmbedder
    from infra.repo.vector_repo import VectorStoreRepo
    from infra.repo.persistent_vector_repo import PersistentVectorRepo
    from infra.repo.graph_repo import GraphRepo
    from infra.repo.persistent_fact_repo import PersistentFactRepo
    from infra.repo.decision_repo import DecisionRepo, PersistentDecisionRepo
    from infra.repo.experience_repo import ExperienceRepo, PersistentExperienceRepo
    from infra.repo.review_repo import PersistentReviewQueueRepo, ReviewQueueRepo
    from infra.repo.cognitive_repo import (
        FailurePatternRepo,
        PersistentFailurePatternRepo,
        PersistentSkillRepo,
        SkillRepo,
    )

    memory_dir = Path(".omni-memory")
    memory_dir.mkdir(parents=True, exist_ok=True)

    if embedder is None:
        if embedding_provider in {"hash", "auto"}:
            selected_embedder = HashEmbedder()
            normalized_provider = "hash"
            normalized_model = "hash"
        else:
            selected_embedder = _build_cli_embedder(
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_device=embedding_device,
            )
            normalized_provider = embedding_provider
            normalized_model = embedding_model or "default"
    else:
        selected_embedder = embedder
        normalized_provider = embedding_provider
        normalized_model = embedding_model or "custom"

    graph_repo = PersistentFactRepo(
        inner=GraphRepo(),
        path=memory_dir / "facts.json",
    )

    decision_repo = PersistentDecisionRepo(
        inner=DecisionRepo(),
        path=memory_dir / "decisions.json",
    )

    experience_repo = PersistentExperienceRepo(
        inner=ExperienceRepo(),
        path=memory_dir / "experiences.json",
    )

    skill_repo = PersistentSkillRepo(
        inner=SkillRepo(),
        path=memory_dir / "skills.json",
    )

    failure_pattern_repo = PersistentFailurePatternRepo(
        inner=FailurePatternRepo(),
        path=memory_dir / "failure_patterns.json",
    )

    review_queue_repo = PersistentReviewQueueRepo(
        inner=ReviewQueueRepo(),
        path=memory_dir / "review_queue.json",
    )

    vector_dir = (
        memory_dir
        / "vector"
        / _safe_name(normalized_provider)
        / _safe_name(normalized_model)
    )

    vector_repo = PersistentVectorRepo(
        inner=VectorStoreRepo(embedder=selected_embedder),
        dir_path=vector_dir,
    )

    return build_memory(
        use_llm=use_llm,
        llm=llm,
        embedder=selected_embedder,
        reject_conflicts=reject_conflicts,
        vector_repo=vector_repo,
        graph_repo=graph_repo,
        decision_repo=decision_repo,
        experience_repo=experience_repo,
        skill_repo=skill_repo,
        failure_pattern_repo=failure_pattern_repo,
        review_queue_repo=review_queue_repo,
    )


def _build_cli_embedder(
    *,
    embedding_provider: str = "hash",
    embedding_model: str | None = None,
    embedding_device: str | None = None,
):
    try:
        if embedding_provider in {"hash", "auto"}:
            return build_embedder(
                backend="hash",
                model_name=embedding_model,
                device=embedding_device,
            )

        return build_embedder(
            backend=embedding_provider,
            model_name=embedding_model,
            device=embedding_device,
        )
    except Exception as exc:
        typer.secho(
            f"Failed to initialize embedder provider={embedding_provider!r}.",
            fg=typer.colors.RED,
            err=True,
        )

        if embedding_provider in {
            "sentence-transformers",
            "sentence_transformers",
            "st",
        }:
            typer.echo(
                "Install sentence-transformers or use --embedding-provider hash.",
                err=True,
            )

        typer.echo(f"Original error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("load-facts")
def load_facts(path: Path = typer.Argument(..., exists=True, readable=True)):
    """Локально загрузить facts JSON через WriteBackService."""
    items = _load_json_list(path)
    rep = _svc().write_items(items)
    _print_report("facts", rep)


@app.command("load-notes")
def load_notes(
    path: Path = typer.Argument(..., exists=True, readable=True),
    embedding_provider: str = typer.Option(
        "hash",
        "--embedding-provider",
        help="BYO-Embedder provider: hash, sentence-transformers.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model name, for example sentence-transformers/all-MiniLM-L6-v2.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional embedding device, for example cpu or cuda.",
    ),
):
    """Локально загрузить notes markdown через WriteBackService."""
    items = _parse_notes_md(path)

    memory = _local_memory(
        use_llm=False,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )

    rep = memory.write_items(items)
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
    k: int = typer.Option(5, "--k"),
    out_json: bool = typer.Option(False, "--json"),
    embedding_provider: str = typer.Option(
        "hash",
        "--embedding-provider",
        help="BYO-Embedder provider: hash, sentence-transformers.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model name, for example sentence-transformers/all-MiniLM-L6-v2.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional embedding device, for example cpu or cuda.",
    ),
):
    memory = _local_memory(
        use_llm=False,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )

    bundle = memory.retrieve(query, k_sem=k)

    if out_json:
        typer.echo(json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    typer.echo(bundle)


@app.command("ask")
def ask_cmd(
    question: str = typer.Argument(...),
    use_llm: bool = typer.Option(False, help="Use configured LLM provider locally."),

    llm_provider: str | None = typer.Option(
        None,
        help="BYO-LLM provider: openai-compatible, openai, ollama, none.",
    ),
    llm_base_url: str | None = typer.Option(
        None,
        help="BYO-LLM base URL, for example http://localhost:11434/v1.",
    ),
    llm_model: str | None = typer.Option(
        None,
        help="BYO-LLM model name.",
    ),
    llm_api_key: str | None = typer.Option(
        None,
        help="BYO-LLM API key; use local/EMPTY for local servers.",
    ),

    embedding_provider: str = typer.Option(
        "auto",
        "--embedding-provider",
        help="BYO-Embedder provider: auto, hash, sentence-transformers.",
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        help="Embedding model name for sentence-transformers.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional embedding device, for example cpu or cuda.",
    ),

    lang: str = typer.Option("en"),
    style: str = typer.Option("concise"),
    out_json: bool = typer.Option(False, "--json"),
):
    """Локально задать вопрос через OmniMemory с optional BYO-LLM/BYO-Embedder."""

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

    try:
        embedder = _build_cli_embedder(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
        )
    except Exception as exc:
        typer.secho(
            f"Failed to initialize embedder provider={embedding_provider!r}.",
            fg=typer.colors.RED,
            err=True,
        )

        if embedding_provider in {"sentence-transformers", "sentence_transformers", "st"}:
            typer.echo(
                "Install sentence-transformers or use --embedding-provider hash.",
                err=True,
            )

        typer.echo(f"Original error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    try:

        answer = _local_memory(
            use_llm=use_llm,
            llm=llm,
            embedder=embedder,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
        ).ask(
            question,
            lang=lang,
            style=style,
        )
    except Exception as exc:
        if llm_provider:
            typer.secho(
                "LLM call failed.",
                fg=typer.colors.RED,
                err=True,
            )

            if llm_base_url:
                typer.echo(f"Provider URL: {llm_base_url}", err=True)

                if "1234" in llm_base_url:
                    typer.echo(
                        "For Ollama OpenAI-compatible API, try: http://localhost:11434/v1",
                        err=True,
                    )

            typer.echo(f"Original error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

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


@app.command("llm-check")
def llm_check_cmd(
    llm_provider: str = typer.Option("openai-compatible"),
    llm_base_url: str = typer.Option("http://localhost:11434/v1"),
    llm_model: str = typer.Option("gemma3:1b"),
    llm_api_key: str = typer.Option("local"),
):
    """Проверить подключение к BYO-LLM провайдеру."""
    from infra.llm.llm_openai_compatible import OpenAICompatibleLLM

    if llm_provider not in {"openai-compatible", "openai_compatible"}:
        typer.secho(f"Unsupported provider: {llm_provider}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    llm = OpenAICompatibleLLM(
        base_url=llm_base_url,
        api_key=llm_api_key,
        model=llm_model,
    )

    try:
        result = llm.generate(
            [{"role": "user", "content": 'Say "ok".'}],
            temperature=0.0,
        )
    except Exception as exc:
        _friendly_llm_error(exc, llm_base_url)

    typer.echo("LLM provider: ok")
    typer.echo(f"model: {result.get('model')}")
    typer.echo(f"text: {result.get('text', '').strip()}")


@app.command("memory-path")
def memory_path_cmd(
    embedding_provider: str = typer.Option("hash", "--embedding-provider"),
    embedding_model: str | None = typer.Option(None, "--embedding-model"),
):
    provider = "hash" if embedding_provider in {"hash", "auto"} else embedding_provider
    model = "hash" if provider == "hash" else embedding_model or "default"

    vector_dir = (
        Path(".omni-memory")
        / "vector"
        / _safe_name(provider)
        / _safe_name(model)
    )

    typer.echo(f"facts:  {Path('.omni-memory') / 'facts.json'}")
    typer.echo(f"vector: {vector_dir}")


@app.command("mcp")
def mcp_cmd(
    use_llm: bool = typer.Option(False, help="Use configured LLM provider for omni_memory_ask."),
    embedding_provider: str = typer.Option(
        "hash",
        "--embedding-provider",
        help="BYO-Embedder provider: hash, sentence-transformers.",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Embedding model name, for example sentence-transformers/all-MiniLM-L6-v2.",
    ),
    embedding_device: str | None = typer.Option(
        None,
        "--embedding-device",
        help="Optional embedding device, for example cpu or cuda.",
    ),
):
    """Run OmniMemory as an MCP stdio server."""
    from app.mcp_server import serve_stdio

    memory = _local_memory(
        use_llm=use_llm,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
    )
    serve_stdio(memory)

if __name__ == "__main__":
    app()

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from omni_memory.cli import app


runner = CliRunner()


def _json_output(result) -> dict:
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_load_facts_outputs_human_text(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps(
            [
                {
                    "id": "fact-cli-alice",
                    "type": "fact",
                    "subject": "alice",
                    "predicate": "at",
                    "object": "lighthouse",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["load-facts", str(facts)])

    assert result.exit_code == 0, result.output
    assert "facts: saved=1 rejected=0" in result.output


def test_write_note_and_retrieve_json_for_automation(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    written = _json_output(
        runner.invoke(
            app,
            [
                "write-note",
                "CLI memory should return structured retrieval JSON.",
                "--output",
                "json",
            ],
        )
    )
    assert written == {"kind": "note", "saved": 1, "rejected": 0, "reasons": []}

    retrieved = _json_output(
        runner.invoke(
            app,
            [
                "retrieve",
                "structured retrieval JSON",
                "--k",
                "3",
                "--output",
                "json",
            ],
        )
    )

    assert retrieved["semantic_chunks"]
    assert retrieved["semantic_chunks"][0]["payload"]["text"] == "CLI memory should return structured retrieval JSON."


def test_load_notes_supports_json_output(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("# CLI heading\n\nA useful CLI note.", encoding="utf-8")

    payload = _json_output(runner.invoke(app, ["load-notes", str(notes), "--output", "json"]))

    assert payload["kind"] == "notes"
    assert payload["saved"] == 2
    assert payload["rejected"] == 0


def test_ask_without_llm_can_emit_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    payload = _json_output(
        runner.invoke(
            app,
            [
                "ask",
                "What do we know?",
                "--embedding-provider",
                "hash",
                "--output",
                "json",
            ],
        )
    )

    assert payload["answer"] == "LLM provider is not configured. Use retrieve/build_context or pass use_llm=True."
    assert payload["model"] is None
    assert payload["used_sections"] == []


def test_memory_path_has_text_and_json_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    text = runner.invoke(app, ["memory-path"])
    assert text.exit_code == 0, text.output
    assert "facts:" in text.output
    assert "vector:" in text.output

    payload = _json_output(runner.invoke(app, ["memory-path", "--output", "json"]))
    assert payload["facts"] == ".omni-memory\\facts.json" or payload["facts"] == ".omni-memory/facts.json"
    assert "vector" in payload


def test_invalid_output_format_fails_with_clear_message(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["memory-path", "--output", "xml"])

    assert result.exit_code != 0
    assert "Expected output to be one of: text, json" in result.output

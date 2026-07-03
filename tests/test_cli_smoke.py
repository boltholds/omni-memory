from pathlib import Path
import json
import subprocess
import sys
import uuid

def _run(args):
    # poetry run python -m app.cli ...   (чтобы не зависеть от entry point)
    return subprocess.run([sys.executable, "-m", "app.cli", *args], capture_output=True, text=True, check=True)

def test_load_facts_and_notes(tmp_path: Path):
    facts = tmp_path / "facts.json"
    fact_id = f"f-{uuid.uuid4().hex}"
    subject = f"alice-{uuid.uuid4().hex}"
    facts.write_text(json.dumps([{"id":fact_id,"subject":subject,"predicate":"at","object":"lighthouse"}]), encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_text("# Hello\nSimple note", encoding="utf-8")

    out1 = _run(["load-facts", str(facts)])
    assert "saved=1" in out1.stdout

    out2 = _run(["load-notes", str(notes)])
    assert "saved=" in out2.stdout

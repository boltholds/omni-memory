from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def evaluate_repo_contract(case: dict[str, Any], parsed: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    repo = case.get("repo")
    if not isinstance(repo, dict):
        return {"applicable": False, "passed": None, "errors": [], "stdout": "", "stderr": ""}

    files = dict(repo.get("files") or {})
    patches = parsed.get("patches") or []
    if not isinstance(patches, list):
        return {"applicable": True, "passed": False, "errors": ["patches_not_list"], "stdout": "", "stderr": ""}

    with tempfile.TemporaryDirectory(prefix="omni-dev-agent-") as tmp:
        root = Path(tmp)
        for rel_path, content in files.items():
            path = root / str(rel_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")

        errors = _apply_patches(root, patches)
        if errors:
            return {"applicable": True, "passed": False, "errors": errors, "stdout": "", "stderr": ""}

        command = repo.get("test_command") or [sys.executable, "-m", "pytest", "-q"]
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            return {"applicable": True, "passed": False, "errors": ["invalid_test_command"], "stdout": "", "stderr": ""}
        if command and command[0] in {"python", "python3"}:
            command = [sys.executable, *command[1:]]

        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "applicable": True,
            "passed": completed.returncode == 0,
            "errors": [] if completed.returncode == 0 else [f"pytest_exit={completed.returncode}"],
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }


def _apply_patches(root: Path, patches: list[Any]) -> list[str]:
    errors: list[str] = []
    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"patch_{idx}_not_object")
            continue
        rel_path = str(patch.get("path") or "")
        find = str(patch.get("find") or "")
        replace = str(patch.get("replace") or "")
        path = (root / rel_path).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"patch_{idx}_escapes_repo:{rel_path}")
            continue
        if not path.exists():
            errors.append(f"patch_{idx}_unknown_path:{rel_path}")
            continue
        if not find:
            errors.append(f"patch_{idx}_missing_find")
            continue
        text = path.read_text(encoding="utf-8")
        if find not in text:
            errors.append(f"patch_{idx}_find_not_found:{rel_path}")
            continue
        path.write_text(text.replace(find, replace, 1), encoding="utf-8")
    return errors

from __future__ import annotations

from typing import Any


def evaluate_patch_contract(case: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    fixture = case.get("fixture")
    if not isinstance(fixture, dict):
        return {"applicable": False, "passed": None, "errors": [], "files": {}}

    files = dict(fixture.get("files") or {})
    patches = parsed.get("patches") or []
    errors: list[str] = []

    if not isinstance(patches, list):
        return {"applicable": True, "passed": False, "errors": ["patches_not_list"], "files": files}

    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"patch_{idx}_not_object")
            continue
        path = str(patch.get("path") or "")
        find = str(patch.get("find") or "")
        replace = str(patch.get("replace") or "")
        if path not in files:
            errors.append(f"patch_{idx}_unknown_path:{path}")
            continue
        if not find:
            errors.append(f"patch_{idx}_missing_find")
            continue
        if find not in files[path]:
            errors.append(f"patch_{idx}_find_not_found:{path}")
            continue
        files[path] = files[path].replace(find, replace, 1)

    for check in fixture.get("checks") or []:
        if not isinstance(check, dict):
            continue
        path = str(check.get("path") or "")
        content = files.get(path, "")
        if "contains" in check and str(check["contains"]) not in content:
            errors.append(f"missing:{path}:{check['contains']}")
        if "not_contains" in check and str(check["not_contains"]) in content:
            errors.append(f"forbidden:{path}:{check['not_contains']}")

    return {
        "applicable": True,
        "passed": not errors,
        "errors": errors,
        "files": files,
    }

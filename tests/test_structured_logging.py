from __future__ import annotations

import json
import logging

from omni_memory.logging import JsonFormatter


def test_json_formatter_includes_structured_extra_fields():
    record = logging.LogRecord(
        name="app.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=10,
        msg="fallback_used",
        args=(),
        exc_info=None,
    )
    record.component = "PromptRenderer"
    record.op = "render_jinja_prompt"
    record.error_type = "UndefinedError"
    record.fallback = "builtin_prompt"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["msg"] == "fallback_used"
    assert payload["component"] == "PromptRenderer"
    assert payload["op"] == "render_jinja_prompt"
    assert payload["error_type"] == "UndefinedError"
    assert payload["fallback"] == "builtin_prompt"

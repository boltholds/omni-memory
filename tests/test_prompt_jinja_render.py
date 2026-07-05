from omni_memory.prompting import PromptRenderer
import logging

def test_jinja_render_fallback_and_basic():
    pr = PromptRenderer(template_dir="templates/prompt")
    msgs = pr.make_messages(
        user_q="Where is Alice?",
        context_sections=["Facts:\n- alice at lighthouse"],
        lang="en",
        style="concise"
    )
    assert msgs[0]["role"] == "system"
    assert "Use ONLY the provided context" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "Where is Alice?" in msgs[1]["content"]


def test_prompt_includes_memory_grounded_contract_when_context_exists():
    pr = PromptRenderer(template_dir="templates/prompt")
    msgs = pr.make_messages(
        user_q="What should we do?",
        context_sections=["Failure Patterns:\n- Do not add unused dependencies."],
        lang="en",
        style="concise",
    )

    assert "Memory-grounded answer contract" in msgs[0]["content"]
    assert "Do not recommend an action that repeats a remembered mistake" in msgs[0]["content"]


def test_prompt_can_disable_memory_grounded_contract():
    pr = PromptRenderer(template_dir="templates/prompt")
    msgs = pr.make_messages(
        user_q="What should we do?",
        context_sections=["Failure Patterns:\n- Do not add unused dependencies."],
        lang="en",
        style="concise",
        extra={"disable_memory_grounding_contract": True},
    )

    assert "Memory-grounded answer contract" not in msgs[0]["content"]


def test_prompt_template_failure_falls_back_and_logs_warning(tmp_path, caplog):
    (tmp_path / "system.j2").write_text("{{ missing_required_value }}", encoding="utf-8")
    (tmp_path / "user.j2").write_text("Question: {{ q }}", encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="app.prompting")

    pr = PromptRenderer(template_dir=str(tmp_path), system_template="system.j2", user_template="user.j2")
    msgs = pr.make_messages("Where is Alice?", ["Facts:\n- alice at lighthouse"])

    assert "Use ONLY the provided context" in msgs[0]["content"]
    assert any(record.message == "prompt_template_render_failed" for record in caplog.records)
    warning = next(record for record in caplog.records if record.message == "prompt_template_render_failed")
    assert warning.op == "render_jinja_prompt"
    assert warning.fallback == "builtin_prompt"

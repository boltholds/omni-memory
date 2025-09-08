from app.prompting import PromptRenderer

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

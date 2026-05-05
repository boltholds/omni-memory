from domain.llm import ILLMProvider, Msg, LLMResult
from app.prompting import PromptRenderer

class DummyLLM(ILLMProvider):
    def generate(self, messages: list[Msg], temperature: float = 0.3) -> LLMResult:
        # echo последнего user-сообщения
        last = [m for m in messages if m["role"] == "user"][-1]["content"]
        return {"text": f"ECHO: {last}", "model": "dummy", "finish_reason": "stop"}

def test_make_messages_and_dummy_llm():
    pr = PromptRenderer(template_dir="templates/prompt")
    msgs = pr.make_messages(
        user_q="Where is Alice?",
        context_sections=["Facts:\n- alice at lighthouse"],
        lang="en",
        style="concise"
    )
    prov = DummyLLM()
    res = prov.generate(msgs)
    assert "ECHO:" in res["text"]

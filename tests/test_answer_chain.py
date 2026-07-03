from __future__ import annotations

from app.config import settings
from app.services.answer_chain import AnswerChainRequest, LangChainAnswerPipeline
from domain.llm import ILLMProvider, LLMResult, Msg
from domain.models import ContextPack, ContextSection, Fact, Provenance, RetrievalBundle
from infra.consistency import SimpleConsistencyEngine


class DummyOrchestrator:
    def __init__(self, bundle: RetrievalBundle, pack: ContextPack | None = None) -> None:
        self.bundle = bundle
        self.pack = pack or ContextPack(sections=[ContextSection(title="Facts", body="- alice at lighthouse")])
        self.queries: list[str] = []
        self.seen_context_budgets: list[int] = []

    def plan_retrieval(self, query: str) -> RetrievalBundle:
        self.queries.append(query)
        return self.bundle

    def assemble_context(self, bundle: RetrievalBundle) -> ContextPack:
        self.seen_context_budgets.append(settings.context_max_tokens)
        return self.pack


class BudgetAwareOrchestrator(DummyOrchestrator):
    def assemble_context(self, bundle: RetrievalBundle) -> ContextPack:
        self.seen_context_budgets.append(settings.context_max_tokens)
        return ContextPack(
            sections=[
                ContextSection(
                    title="Budget",
                    body=f"- context budget {settings.context_max_tokens}",
                )
            ]
        )


class EchoLLM(ILLMProvider):
    def __init__(self) -> None:
        self.temperature: float | None = None

    def generate(self, messages: list[Msg], temperature: float = 0.3) -> LLMResult:
        self.temperature = temperature
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        return {"text": f"ECHO: {user}", "model": "echo", "finish_reason": "stop"}


class FailingLLM(ILLMProvider):
    def generate(self, messages: list[Msg], temperature: float = 0.3) -> LLMResult:
        raise TimeoutError("offline")


class PromptSpy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def make_messages(self, user_q: str, context_sections: list[str], lang: str, style: str):
        self.calls.append({"user_q": user_q, "context_sections": context_sections, "lang": lang, "style": style})
        return [{"role": "user", "content": f"{user_q}\n{context_sections}"}]


def _fact(fid: str, obj: str) -> Fact:
    return Fact(id=fid, subject="alice", predicate="at", object=obj, provenance=Provenance(source="test"))


def test_answer_chain_generates_with_context_and_style_mapping():
    orchestrator = DummyOrchestrator(RetrievalBundle(facts=[_fact("f1", "lighthouse")]))
    prompt = PromptSpy()
    llm = EchoLLM()
    pipeline = LangChainAnswerPipeline(
        orchestrator=orchestrator,
        consistency=SimpleConsistencyEngine(),
        llm_provider=llm,
        prompt_renderer=prompt,
    )

    result = pipeline.run(AnswerChainRequest(q="Where is Alice?", lang="ru", style="plain", temperature=0.1))

    assert result.model == "echo"
    assert result.used_sections == ["Facts"]
    assert "ECHO:" in result.answer
    assert orchestrator.queries == ["Where is Alice?"]
    assert prompt.calls[0]["style"] == "concise"
    assert prompt.calls[0]["lang"] == "ru"
    assert llm.temperature == 0.1


def test_answer_chain_reports_conflicts_and_llm_failure():
    bundle = RetrievalBundle(facts=[_fact("f1", "lighthouse"), _fact("f2", "bridge")])
    pipeline = LangChainAnswerPipeline(
        orchestrator=DummyOrchestrator(bundle),
        consistency=SimpleConsistencyEngine(),
        llm_provider=FailingLLM(),
        prompt_renderer=PromptSpy(),
    )

    result = pipeline.run(AnswerChainRequest(q="Where is Alice?"))

    assert result.answer.startswith("Conflict detected.")
    assert "LLM provider failed" in result.answer
    assert any("Detected 1 conflict" in advisory for advisory in result.advisories)
    assert "LLM provider failed: TimeoutError" in result.advisories


def test_answer_chain_without_llm_returns_configured_message():
    pipeline = LangChainAnswerPipeline(
        orchestrator=DummyOrchestrator(RetrievalBundle(), ContextPack()),
        consistency=SimpleConsistencyEngine(),
        llm_provider=None,
        prompt_renderer=PromptSpy(),
    )

    result = pipeline.run(AnswerChainRequest(q="Anything?"))

    assert result.answer == "LLM provider is not configured (LLM_PROVIDER=none)."
    assert result.model is None
    assert result.used_sections == []


def test_answer_chain_honors_request_context_budget_in_generated_prompt(monkeypatch):
    orchestrator = BudgetAwareOrchestrator(RetrievalBundle())
    prompt = PromptSpy()
    monkeypatch.setattr(settings, "context_max_tokens", 123)
    pipeline = LangChainAnswerPipeline(
        orchestrator=orchestrator,
        consistency=SimpleConsistencyEngine(),
        llm_provider=EchoLLM(),
        prompt_renderer=prompt,
    )

    result = pipeline.run(AnswerChainRequest(q="budget?", max_tokens=7))

    assert result.used_sections == ["Budget"]
    assert "context budget 7" in prompt.calls[0]["context_sections"][0]

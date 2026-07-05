from __future__ import annotations

from omni_memory.services.simple_answer_pipeline import (
    AnswerPipelineRequest,
    SimpleAnswerPipeline,
)
from omni_memory.domain.models import ContextPack, ContextSection


class AddContextStrategy:
    name = "add_context"

    def handle(self, state):
        return {
            **state,
            "pack": ContextPack(
                sections=[ContextSection(title="Facts", body="OmniMemory stores governed memory.")],
                advisories=["test_advisory"],
            ),
        }


class AddAnswerStrategy:
    name = "add_answer"

    def handle(self, state):
        return {**state, "answer_text": "Governed memory.", "model": "fake-model"}


def test_simple_answer_pipeline_runs_dependency_light_strategies():
    pipeline = SimpleAnswerPipeline(
        orchestrator=object(),
        consistency=object(),
        llm_provider=None,
        prompt_renderer=object(),
        strategies=[AddContextStrategy(), AddAnswerStrategy()],
    )

    result = pipeline.run(AnswerPipelineRequest(q="What is OmniMemory?"))

    assert result.answer == "Governed memory."
    assert result.model == "fake-model"
    assert result.advisories == ["test_advisory"]
    assert result.used_sections == ["Facts"]


def test_simple_answer_pipeline_exposes_neutral_pipeline_attribute():
    pipeline = SimpleAnswerPipeline(
        orchestrator=object(),
        consistency=object(),
        llm_provider=None,
        prompt_renderer=object(),
        strategies=[AddContextStrategy(), AddAnswerStrategy()],
    )

    assert hasattr(pipeline, "pipeline")
    assert not hasattr(pipeline, "chain")

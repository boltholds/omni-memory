from __future__ import annotations

import time
import uuid
import re
from dataclasses import dataclass
from typing import Any

from .embeddings import build_embedder
from .config import settings
from .retriever import Retriever
from .orchestrator import Orchestrator
from .writeback import WriteBackService
from .prompting import PromptRenderer

from domain.policy import MemoryPolicy
from infra.vector_repo import VectorStoreRepo
from infra.graph_repo import GraphRepo
from infra.episodic_repo import EpisodicRepo
from infra.consistency import SimpleConsistencyEngine
from infra.llm_factory import build_llm


@dataclass
class MemoryAnswer:
    answer: str
    advisories: list[str]
    used_sections: list[str]
    context: dict[str, Any]


class OmniMemory:
    def __init__(self, use_llm: bool = False) -> None:
        embedder = build_embedder(settings.embedding_backend, settings.embedding_model)

        self.vector_repo = VectorStoreRepo(embedder=embedder)
        self.graph_repo = GraphRepo()
        self.episodic_repo = EpisodicRepo(db_path=settings.sqlite_path)

        self.retriever = Retriever(
            self.vector_repo,
            self.graph_repo,
            self.episodic_repo,
        )

        self.consistency = SimpleConsistencyEngine()

        self.orchestrator = Orchestrator(
            self.retriever,
            self.consistency,
        )

        self.writeback_service = WriteBackService(
            self.vector_repo,
            self.graph_repo,
            self.episodic_repo,
            MemoryPolicy(),
        )

        self.prompt_renderer = PromptRenderer()
        self.llm = build_llm() if use_llm else None


    def write_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        *,
        source: str = "user",
        confidence: float = 1.0,
    ) -> None:
        fact = {
            "id": f"fact-{uuid.uuid4().hex}",
            "subject": subject.lower().strip(),
            "predicate": predicate.lower().strip(),
            "object": object_.lower().strip(),
            "confidence": confidence,
            "provenance": {
                "source": source,
                "time": time.time(),
                "meta": {},
            },
            "meta": {},
        }

        report = self.writeback_service.write([fact])

        if report.rejected:
            raise RuntimeError(f"Write rejected: {report.reasons}")

    def ask(self, question: str, debug: bool = False) -> MemoryAnswer:
        bundle = self.orchestrator.plan_retrieval(question)
        pack = self.orchestrator.assemble_context(bundle)

        conflict_report = self.consistency.detect_conflicts(bundle.facts)

        if self.llm is None:
            answer = self._fallback_answer(question, bundle, conflict_report)
        else:
            sections = [f"{s.title}:\n{s.body}" for s in pack.sections]
            messages = self.prompt_renderer.make_messages(
                question,
                sections,
                lang="en",
                style="concise",
            )
            result = self.llm.generate(messages)
            answer = result.get("text", "").strip()

        context = {
            "facts": [f.model_dump() for f in bundle.facts],
            "episodes": [e.model_dump() for e in bundle.episodes],
            "semantic_chunks": [s.model_dump() for s in bundle.semantic_chunks],
            "conflicts": [c.model_dump() for c in conflict_report.conflicts],
            "sections": [s.model_dump() for s in pack.sections],
        }

        if debug:
            print("=== FACTS ===")
            for fact in bundle.facts:
                print(f"- {fact.subject} {fact.predicate} {fact.object}")

            print("\n=== CONFLICTS ===")
            for conflict in conflict_report.conflicts:
                print(f"- {conflict.key}: {', '.join(conflict.variants)}")

            print("\n=== ANSWER ===")
            print(answer)

        return MemoryAnswer(
            answer=answer,
            advisories=pack.advisories,
            used_sections=[s.title for s in pack.sections],
            context=context,
        )

    def _text_to_memory_item(self, text: str) -> dict[str, Any]:
        parsed_fact = self._try_parse_demo_fact(text)

        if parsed_fact is not None:
            return parsed_fact

        return {
            "id": f"note-{uuid.uuid4().hex}",
            "type": "note",
            "payload": {"text": text},
            "provenance": {
                "source": "user",
                "time": time.time(),
                "meta": {},
            },
            "meta": {},
        }

    def _try_parse_demo_fact(self, text: str) -> dict[str, Any] | None:
        normalized = text.strip().lower()

        patterns = [
            r"^(?P<subject>\w+)\s+lives\s+in\s+(?:the\s+)?(?P<object>[\w\s-]+)$",
            r"^(?P<subject>\w+)\s+moved\s+to\s+(?:the\s+)?(?P<object>[\w\s-]+)$",
            r"^(?P<subject>\w+)\s+is\s+at\s+(?:the\s+)?(?P<object>[\w\s-]+)$",
            r"^(?P<subject>\w+)\s+works\s+with\s+(?P<object>[\w\s-]+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, normalized)
            if not match:
                continue

            subject = match.group("subject").strip()
            obj = match.group("object").strip()

            predicate = "at"
            if "works with" in normalized:
                predicate = "works_with"

            return {
                "id": f"fact-{uuid.uuid4().hex}",
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "provenance": {
                    "source": "user",
                    "time": time.time(),
                    "meta": {"raw_text": text},
                },
                "meta": {},
            }

        return None

    def _fallback_answer(self, question, bundle, conflict_report) -> str:
        if conflict_report.conflicts:
            conflict = conflict_report.conflicts[0]
            variants = conflict.variants

            latest_fact = None
            for fact in bundle.facts:
                if f"{fact.subject}::{fact.predicate}" == conflict.key:
                    latest_fact = fact

            selected = latest_fact.object if latest_fact else variants[-1]

            return (
                f"{selected} is the most likely answer.\n\n"
                f"However, conflicting memory was found:\n"
                f"- {', '.join(variants)}\n\n"
                f"Selected: {selected}."
            )

        if bundle.facts:
            fact = bundle.facts[-1]
            return f"{fact.subject} {fact.predicate} {fact.object}."

        if bundle.semantic_chunks:
            text = bundle.semantic_chunks[0].payload.get("text", "")
            return f"I found this in memory: {text}"

        return "I do not have enough memory context to answer."
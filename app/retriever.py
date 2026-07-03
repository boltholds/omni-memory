# app/retriever.py
from __future__ import annotations

from typing import List, Set

from domain.models import RetrievalBundle, Fact, Episode, DecisionRecord
from domain.ports import IRetriever, IMemoryReadRepository, IGraphRepository, IEpisodicRepository, IDecisionRepository
from app.entities import build_entity_stack
from app.config import settings
from app.profiling import timed
from app.stats import stats
from infra.consistency import build_fact_beliefs


def _simple_entities(query: str) -> List[str]:
    """
    Примитивная NER-заглушка:
    - токены из букв/цифр длиной >= 3
    - без дубликатов, регистр игнорируем
    """
    seen: Set[str] = set()
    ents: List[str] = []
    for raw in query.replace("_", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum()).lower()
        if len(tok) >= 3 and tok not in seen:
            seen.add(tok)
            ents.append(tok)
    return ents


def _entity_variants(entity: str) -> list[str]:
    """Return stable lookup variants for graph nodes.

    Graph facts can come from different write paths: some values are already
    normalized to lowercase, while object values may preserve original casing.
    Retrieval should try both forms so a first-hop object like "OmniMemory" can
    lead to a second-hop subject like "omnimemory".
    """

    value = str(entity or "").strip()
    if not value:
        return []

    variants = [value, value.lower(), value.casefold()]
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _compound_entities(entities: list[str], *, max_n: int = 3) -> list[str]:
    """Add simple adjacent n-grams for graph keys such as memory_project."""

    out: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value and value not in seen:
            seen.add(value)
            out.append(value)

    for entity in entities:
        add(entity)

    for n in range(2, max_n + 1):
        if len(entities) < n:
            continue
        for i in range(0, len(entities) - n + 1):
            window = entities[i : i + n]
            add("_".join(window))

    return out


def _add_fact(fact: Fact, facts: list[Fact], seen_ids: set[str]) -> bool:
    if fact.id in seen_ids:
        return False
    seen_ids.add(fact.id)
    facts.append(fact)
    return True


class Retriever(IRetriever):
    """
    Объединённый извлекатель:
      - semantic: векторный поиск по исходному запросу
      - graph: выборка фактов по найденным сущностям (subject/object)
      - graph 2-hop: расширение по subject/object найденных фактов
      - episodic: поиск эпизодов по сущностям
    """

    def __init__(
        self,
        vector_repo: IMemoryReadRepository,
        graph_repo: IGraphRepository,
        episodic_repo: IEpisodicRepository,
        decision_repo: IDecisionRepository | None = None,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo
        self._decisions = decision_repo
        self._extractor, self._linker = build_entity_stack(settings.ner_backend, settings.entity_aliases)

    def _query_graph_entity(self, entity: str, facts: list[Fact], seen_ids: set[str]) -> list[Fact]:
        found: list[Fact] = []
        for variant in _entity_variants(entity):
            for fact in self._graph.query(subject=variant):
                if _add_fact(fact, facts, seen_ids):
                    found.append(fact)
            for fact in self._graph.query(object=variant):
                if _add_fact(fact, facts, seen_ids):
                    found.append(fact)
        return found

    def _expand_graph_two_hop(self, seed_entities: list[str]) -> list[Fact]:
        facts: list[Fact] = []
        seen_ids: set[str] = set()

        frontier: list[str] = list(seed_entities)
        seen_entities: set[str] = set()

        for _hop in range(2):
            next_frontier: list[str] = []

            for entity in frontier:
                variants = _entity_variants(entity)
                if not variants:
                    continue

                canonical = variants[-1]
                if canonical in seen_entities:
                    continue
                seen_entities.add(canonical)

                found = self._query_graph_entity(entity, facts, seen_ids)
                for fact in found:
                    next_frontier.extend([fact.subject, fact.object])

            frontier = next_frontier
            if not frontier:
                break

        return facts

    @timed("retriever.retrieve", slow_ms=100)
    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        raw_ents = self._extractor.extract(query)
        linked_ents = self._linker.link_all(raw_ents)
        ents = _compound_entities(linked_ents)

        # I Семантические чанки
        stop_vec = stats.timeit("retriever.vec_ms")
        semantic_chunks = self._vector.semantic_search(query, k=k_sem)
        stop_vec()

        # II Факты: прямой поиск + 2-hop expansion по графу.
        stop_kg = stats.timeit("retriever.kg_ms")
        facts: List[Fact] = self._expand_graph_two_hop(ents)
        stop_kg()

        # III Эпизоды (пользователя пока не извлекаем -> None)
        stop_ep = stats.timeit("retriever.ep_ms")
        episodes: List[Episode] = self._episodic.search(user=None, entities=ents, k=k_eps)
        stop_ep()

        decisions: List[DecisionRecord] = []
        if self._decisions is not None:
            decisions = self._decisions.search(query, k=k_eps)

        beliefs = build_fact_beliefs(facts)

        return RetrievalBundle(
            semantic_chunks=semantic_chunks,
            facts=facts,
            beliefs=beliefs,
            episodes=episodes,
            decisions=decisions,
            citations=[],
        )

from __future__ import annotations

from typing import List, Set

from app.config import settings
from app.entities import build_entity_stack
from app.memory_planner import MemoryPlanner
from app.profiling import timed
from app.stats import stats
from domain.models import DecisionRecord, Episode, ExperienceRecord, Fact, RetrievalBundle, SkillRecord
from domain.models import FailurePatternRecord as PatternRecord
from domain.ports import IDecisionRepository, IEpisodicRepository, IExperienceRepository, IGraphRepository, IMemoryReadRepository, IRetriever, ISkillRepository
from domain.ports import IFailurePatternRepository as PatternRepository
from infra.consistency import build_fact_beliefs


def _simple_entities(query: str) -> List[str]:
    seen: Set[str] = set()
    ents: List[str] = []
    for raw in query.replace("_", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum()).lower()
        if len(tok) >= 3 and tok not in seen:
            seen.add(tok)
            ents.append(tok)
    return ents


def _entity_variants(entity: str) -> list[str]:
    value = str(entity or "").strip()
    if not value:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in [value, value.lower(), value.casefold()]:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _compound_entities(entities: list[str], *, max_n: int = 3) -> list[str]:
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
            add("_".join(entities[i : i + n]))
    return out


def _add_fact(fact: Fact, facts: list[Fact], seen_ids: set[str]) -> bool:
    if fact.id in seen_ids:
        return False
    seen_ids.add(fact.id)
    facts.append(fact)
    return True


class Retriever(IRetriever):
    def __init__(
        self,
        vector_repo: IMemoryReadRepository,
        graph_repo: IGraphRepository,
        episodic_repo: IEpisodicRepository,
        decision_repo: IDecisionRepository | None = None,
        experience_repo: IExperienceRepository | None = None,
        skill_repo: ISkillRepository | None = None,
        pattern_repo: PatternRepository | None = None,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo
        self._decisions = decision_repo
        self._experiences = experience_repo
        self._skills = skill_repo
        self._patterns = pattern_repo
        self._extractor, self._linker = build_entity_stack(settings.ner_backend, settings.entity_aliases)
        self._planner = MemoryPlanner()

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
    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3, intent: str | None = None, mode: str | None = None) -> RetrievalBundle:
        profile = self._planner.profile(intent, mode=mode)
        raw_ents = self._extractor.extract(query)
        linked_ents = self._linker.link_all(raw_ents)
        ents = _compound_entities(linked_ents)

        semantic_chunks = []
        if profile.semantic:
            stop_vec = stats.timeit("retriever.vec_ms")
            semantic_chunks = self._vector.semantic_search(query, k=k_sem)
            stop_vec()

        facts: List[Fact] = []
        if profile.facts:
            stop_kg = stats.timeit("retriever.kg_ms")
            facts = self._expand_graph_two_hop(ents)
            stop_kg()

        episodes: List[Episode] = []
        if profile.episodes:
            stop_ep = stats.timeit("retriever.ep_ms")
            episodes = self._episodic.search(user=None, entities=ents, k=k_eps)
            stop_ep()

        decisions: List[DecisionRecord] = []
        if profile.decisions and self._decisions is not None:
            decisions = self._decisions.search(query, k=k_eps)

        experiences: List[ExperienceRecord] = []
        if profile.experiences and self._experiences is not None:
            experiences = self._experiences.search(query, k=k_eps)

        skills: List[SkillRecord] = []
        if profile.skills and self._skills is not None:
            skills = self._skills.search(query, k=k_eps)

        patterns: List[PatternRecord] = []
        if profile.failure_patterns and self._patterns is not None:
            patterns = self._patterns.search(query, k=k_eps)

        beliefs = build_fact_beliefs(facts) if profile.beliefs else []
        return RetrievalBundle(
            semantic_chunks=semantic_chunks,
            facts=facts,
            beliefs=beliefs,
            episodes=episodes,
            decisions=decisions,
            experiences=experiences,
            skills=skills,
            failure_patterns=patterns,
            citations=[],
        )

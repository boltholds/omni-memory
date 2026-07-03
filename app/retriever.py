from __future__ import annotations

from typing import Any, List, Set

from app.config import settings
from app.entities import build_entity_stack
from app.memory_planner import MemoryPlanner
from app.profiling import timed
from app.stats import stats
from domain.models import DecisionRecord, Episode, ExperienceRecord, Fact, MemoryObject, RetrievalBundle, SkillRecord
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
        domain_graph_repo: Any | None = None,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo
        self._decisions = decision_repo
        self._experiences = experience_repo
        self._skills = skill_repo
        self._patterns = pattern_repo
        self._domain_graph = domain_graph_repo
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
        domain_weights = _domain_weights(query, self._domain_graph)

        semantic_chunks: list[MemoryObject] = []
        if profile.semantic:
            stop_vec = stats.timeit("retriever.vec_ms")
            semantic_chunks = self._rank_memory_items(
                self._vector.semantic_search(query, k=max(k_sem * 3, k_sem)),
                domain_weights,
                limit=k_sem,
            )
            stop_vec()

        facts: List[Fact] = []
        if profile.facts:
            stop_kg = stats.timeit("retriever.kg_ms")
            facts = self._rank_memory_items(self._expand_graph_two_hop(ents), domain_weights)
            stop_kg()

        episodes: List[Episode] = []
        if profile.episodes:
            stop_ep = stats.timeit("retriever.ep_ms")
            episodes = self._rank_memory_items(
                self._episodic.search(user=None, entities=ents, k=max(k_eps * 3, k_eps)),
                domain_weights,
                limit=k_eps,
            )
            stop_ep()

        decisions: List[DecisionRecord] = []
        if profile.decisions and self._decisions is not None:
            decisions = self._rank_memory_items(self._decisions.search(query, k=max(k_eps * 3, k_eps)), domain_weights, limit=k_eps)

        experiences: List[ExperienceRecord] = []
        if profile.experiences and self._experiences is not None:
            experiences = self._rank_memory_items(self._experiences.search(query, k=max(k_eps * 3, k_eps)), domain_weights, limit=k_eps)

        skills: List[SkillRecord] = []
        if profile.skills and self._skills is not None:
            skills = self._rank_memory_items(self._skills.search(query, k=max(k_eps * 3, k_eps)), domain_weights, limit=k_eps)

        patterns: List[PatternRecord] = []
        if profile.failure_patterns and self._patterns is not None:
            patterns = self._rank_memory_items(self._patterns.search(query, k=max(k_eps * 3, k_eps)), domain_weights, limit=k_eps)

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

    def _rank_memory_items(self, items: list[Any], domain_weights: dict[str, float], *, limit: int | None = None) -> list[Any]:
        ranked = sorted(
            items,
            key=lambda item: (_memory_score(item, domain_weights), _memory_time(item)),
            reverse=True,
        )
        return ranked[:limit] if limit is not None else ranked


def _domain_weights(query: str, domain_graph: Any | None) -> dict[str, float]:
    if domain_graph is None or not hasattr(domain_graph, "list_nodes"):
        return {}

    normalized_query = _normalize_text(query)
    weights: dict[str, float] = {}
    for node in domain_graph.list_nodes():
        labels = [node.id, node.name, *getattr(node, "aliases", [])]
        if any(_domain_label_matches(normalized_query, label) for label in labels):
            weights[node.id] = max(weights.get(node.id, 0.0), 3.0)
            if hasattr(domain_graph, "reachable_domain_ids"):
                for distance, related_id in enumerate(domain_graph.reachable_domain_ids(node.id, max_depth=2), start=1):
                    weights[related_id] = max(weights.get(related_id, 0.0), max(1.0, 2.0 - 0.5 * distance))
    return weights


def _domain_label_matches(normalized_query: str, label: str) -> bool:
    label = str(label or "").strip()
    if not label:
        return False
    candidates = {
        label,
        label.split(":")[-1],
        label.replace("_", "-"),
        label.replace("-", " "),
        label.replace("_", " "),
    }
    return any(_normalize_text(candidate) in normalized_query for candidate in candidates if candidate)


def _memory_score(item: Any, domain_weights: dict[str, float]) -> float:
    score = 0.0
    scope = _scope(item)
    for domain_id in _scope_domain_ids(scope):
        score += domain_weights.get(domain_id, 0.0)

    environment = str(scope.get("environment") or "").lower()
    durability = str(scope.get("durability") or "").lower()
    excluded = bool(scope.get("exclude_from_consolidation") or _meta(item).get("exclude_from_consolidation"))

    if environment in {"test", "benchmark", "sandbox"}:
        score -= 2.0
    if durability in {"ephemeral", "session"}:
        score -= 2.0
    if excluded:
        score -= 1.0
    return score


def _memory_time(item: Any) -> float:
    provenance = getattr(item, "provenance", None)
    if provenance is None:
        return 0.0
    try:
        return float(getattr(provenance, "time", 0.0) or 0.0)
    except Exception:
        return 0.0


def _scope(item: Any) -> dict[str, Any]:
    scope = _meta(item).get("scope") or {}
    if hasattr(scope, "model_dump"):
        return scope.model_dump(mode="json")
    return dict(scope) if isinstance(scope, dict) else {}


def _meta(item: Any) -> dict[str, Any]:
    meta = getattr(item, "meta", {}) or {}
    return dict(meta) if isinstance(meta, dict) else {}


def _scope_domain_ids(scope: dict[str, Any]) -> list[str]:
    raw = scope.get("domain_ids") or []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if str(item)]


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())

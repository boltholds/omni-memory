from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class RetrievalScopeFilter:
    domain_ids: list[str] = field(default_factory=list)
    environments: list[str] = field(default_factory=list)
    durabilities: list[str] = field(default_factory=list)
    memory_types: list[str] = field(default_factory=list)
    include_ephemeral: bool = True
    strict_domains: bool = False
    expand_domains: bool = True

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> "RetrievalScopeFilter":
        if raw is None:
            return cls()
        return cls(
            domain_ids=_string_list(raw.get("domain_ids") or raw.get("domains")),
            environments=[_normalize_token(item) for item in _string_list(raw.get("environments") or raw.get("environment"))],
            durabilities=[_normalize_token(item) for item in _string_list(raw.get("durabilities") or raw.get("durability"))],
            memory_types=[_normalize_memory_type(item) for item in _string_list(raw.get("memory_types") or raw.get("types") or raw.get("type"))],
            include_ephemeral=bool(raw.get("include_ephemeral", True)),
            strict_domains=bool(raw.get("strict_domains", False)),
            expand_domains=bool(raw.get("expand_domains", True)),
        )


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
    def retrieve(
        self,
        query: str,
        k_sem: int = 5,
        k_eps: int = 3,
        intent: str | None = None,
        mode: str | None = None,
        scope: dict[str, Any] | None = None,
    ) -> RetrievalBundle:
        profile = self._planner.profile(intent, mode=mode)
        raw_ents = self._extractor.extract(query)
        linked_ents = self._linker.link_all(raw_ents)
        ents = _compound_entities(linked_ents)
        scope_filter = RetrievalScopeFilter.from_raw(scope)
        domain_weights = _domain_weights(query, self._domain_graph, scope_filter)

        semantic_chunks: list[MemoryObject] = []
        if profile.semantic and _memory_type_allowed("note", scope_filter):
            stop_vec = stats.timeit("retriever.vec_ms")
            semantic_chunks = self._rank_memory_items(
                self._vector.semantic_search(query, k=max(k_sem * 3, k_sem)),
                domain_weights,
                scope_filter,
                memory_type="note",
                limit=k_sem,
            )
            stop_vec()

        facts: List[Fact] = []
        if profile.facts and _memory_type_allowed("fact", scope_filter):
            stop_kg = stats.timeit("retriever.kg_ms")
            facts = self._rank_memory_items(
                self._expand_graph_two_hop(ents),
                domain_weights,
                scope_filter,
                memory_type="fact",
            )
            stop_kg()

        episodes: List[Episode] = []
        if profile.episodes and _memory_type_allowed("episode", scope_filter):
            stop_ep = stats.timeit("retriever.ep_ms")
            episodes = self._rank_memory_items(
                self._episodic.search(user=None, entities=ents, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="episode",
                limit=k_eps,
            )
            stop_ep()

        decisions: List[DecisionRecord] = []
        if profile.decisions and self._decisions is not None and _memory_type_allowed("decision", scope_filter):
            decisions = self._rank_memory_items(
                self._decisions.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="decision",
                limit=k_eps,
            )

        experiences: List[ExperienceRecord] = []
        if profile.experiences and self._experiences is not None and _memory_type_allowed("experience", scope_filter):
            experiences = self._rank_memory_items(
                self._experiences.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="experience",
                limit=k_eps,
            )

        skills: List[SkillRecord] = []
        if profile.skills and self._skills is not None and _memory_type_allowed("skill", scope_filter):
            skills = self._rank_memory_items(
                self._skills.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="skill",
                limit=k_eps,
            )

        patterns: List[PatternRecord] = []
        if profile.failure_patterns and self._patterns is not None and _memory_type_allowed("failure_pattern", scope_filter):
            patterns = self._rank_memory_items(
                self._patterns.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="failure_pattern",
                limit=k_eps,
            )

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

    def _rank_memory_items(
        self,
        items: list[Any],
        domain_weights: dict[str, float],
        scope_filter: RetrievalScopeFilter,
        *,
        memory_type: str,
        limit: int | None = None,
    ) -> list[Any]:
        filtered = [
            item
            for item in items
            if _passes_scope_filter(item, memory_type=memory_type, scope_filter=scope_filter)
        ]
        ranked = sorted(
            filtered,
            key=lambda item: (_memory_score(item, domain_weights), _memory_time(item)),
            reverse=True,
        )
        return ranked[:limit] if limit is not None else ranked


def _domain_weights(query: str, domain_graph: Any | None, scope_filter: RetrievalScopeFilter) -> dict[str, float]:
    weights: dict[str, float] = {domain_id: 4.0 for domain_id in scope_filter.domain_ids}
    if domain_graph is None or not hasattr(domain_graph, "list_nodes"):
        return weights

    if scope_filter.expand_domains and hasattr(domain_graph, "reachable_domain_ids"):
        for source_id in list(weights):
            for distance, related_id in enumerate(domain_graph.reachable_domain_ids(source_id, max_depth=2), start=1):
                weights[related_id] = max(weights.get(related_id, 0.0), max(1.0, 2.0 - 0.5 * distance))

    normalized_query = _normalize_text(query)
    for node in domain_graph.list_nodes():
        labels = [node.id, node.name, *getattr(node, "aliases", [])]
        if any(_domain_label_matches(normalized_query, label) for label in labels):
            weights[node.id] = max(weights.get(node.id, 0.0), 3.0)
            if scope_filter.expand_domains and hasattr(domain_graph, "reachable_domain_ids"):
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


def _passes_scope_filter(item: Any, *, memory_type: str, scope_filter: RetrievalScopeFilter) -> bool:
    if not _memory_type_allowed(memory_type, scope_filter):
        return False

    scope = _scope(item)
    environment = _normalize_token(scope.get("environment") or "")
    durability = _normalize_token(scope.get("durability") or "")

    if scope_filter.environments and environment not in scope_filter.environments:
        return False
    if scope_filter.durabilities and durability not in scope_filter.durabilities:
        return False
    if not scope_filter.include_ephemeral and durability in {"ephemeral", "session"}:
        return False
    if scope_filter.strict_domains and scope_filter.domain_ids:
        item_domains = set(_scope_domain_ids(scope))
        if not item_domains.intersection(scope_filter.domain_ids):
            return False
    return True


def _memory_type_allowed(memory_type: str, scope_filter: RetrievalScopeFilter) -> bool:
    if not scope_filter.memory_types:
        return True
    return _normalize_memory_type(memory_type) in scope_filter.memory_types


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


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [])
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        normalized = str(item).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    return out


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_")


def _normalize_memory_type(value: Any) -> str:
    token = _normalize_token(value)
    aliases = {
        "notes": "note",
        "semantic": "note",
        "semantic_chunk": "note",
        "semantic_chunks": "note",
        "facts": "fact",
        "episodes": "episode",
        "decisions": "decision",
        "experiences": "experience",
        "skills": "skill",
        "failure_patterns": "failure_pattern",
        "failure-pattern": "failure_pattern",
        "failurepattern": "failure_pattern",
    }
    return aliases.get(token, token)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("_", " ").replace("-", " ").split())

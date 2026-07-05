from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, List, Set

from omni_memory.config import settings
from omni_memory.entities import build_entity_stack
from omni_memory.memory_planner import MemoryPlanner
from omni_memory.profiling import timed
from omni_memory.stats import stats
from omni_memory.telemetry import span as telemetry_span
from omni_memory.domain.model_ports import IReranker
from omni_memory.domain.models import DecisionRecord, Episode, ExperienceRecord, Fact, MemoryObject, RetrievalBundle, SkillRecord
from omni_memory.domain.models import FailurePatternRecord as PatternRecord
from omni_memory.domain.ports import IDecisionRepository, IEpisodicRepository, IExperienceRepository, IGraphRepository, IMemoryReadRepository, IRetriever, ISkillRepository
from omni_memory.domain.ports import IFailurePatternRepository as PatternRepository
from omni_memory.infra.consistency import build_fact_beliefs
from omni_memory.infra.rerankers.factory import reranker_candidate_budget

_MAX_GRAPH_FRONTIER = 64
_MAX_GRAPH_FACTS = 512
_MAX_GRAPH_FALLBACK_QUERY_CALLS = 128
_RERANK_POOL_MULTIPLIER = 4
_RERANK_POOL_MIN = 16
_UNSCOPED_DURABLE_PENALTY = 1.25
_INTENT_PRIORITY_STEP = 0.05

_BUNDLE_SECTION_TYPES: dict[str, tuple[str, str]] = {
    "facts": ("facts", "fact"),
    "episodes": ("episodes", "episode"),
    "decisions": ("decisions", "decision"),
    "relevant_experience": ("experiences", "experience"),
    "skills": ("skills", "skill"),
    "failure_patterns": ("failure_patterns", "failure_pattern"),
    "semantic_notes": ("semantic_chunks", "note"),
}
_DEFAULT_BUNDLE_ORDER: tuple[tuple[str, str], ...] = (
    ("facts", "fact"),
    ("episodes", "episode"),
    ("decisions", "decision"),
    ("experiences", "experience"),
    ("skills", "skill"),
    ("failure_patterns", "failure_pattern"),
    ("semantic_chunks", "note"),
)


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


@dataclass
class _GraphExpansionState:
    fallback_query_calls: int = 0
    fallback_truncated: bool = False


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
        reranker: IReranker | None = None,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo
        self._decisions = decision_repo
        self._experiences = experience_repo
        self._skills = skill_repo
        self._patterns = pattern_repo
        self._domain_graph = domain_graph_repo
        self._reranker = reranker
        self._extractor, self._linker = build_entity_stack(settings.ner_backend, settings.entity_aliases)
        self._planner = MemoryPlanner()

    def _query_graph_entities(
        self,
        entities: list[str],
        facts: list[Fact],
        seen_ids: set[str],
        state: _GraphExpansionState,
    ) -> list[Fact]:
        found: list[Fact] = []
        if not entities:
            return found

        if hasattr(self._graph, "query_entity_neighborhood"):
            stats.inc("retriever.graph_neighborhood_calls")
            candidates = self._graph.query_entity_neighborhood(entities)  # type: ignore[attr-defined]
        else:
            candidates = []
            before_calls = state.fallback_query_calls
            for entity in entities:
                if state.fallback_query_calls >= _MAX_GRAPH_FALLBACK_QUERY_CALLS:
                    state.fallback_truncated = True
                    break
                candidates.extend(self._graph.query(subject=entity))
                state.fallback_query_calls += 1
                if state.fallback_query_calls >= _MAX_GRAPH_FALLBACK_QUERY_CALLS:
                    state.fallback_truncated = True
                    break
                candidates.extend(self._graph.query(object=entity))
                state.fallback_query_calls += 1
            stats.inc("retriever.graph_fallback_query_calls", state.fallback_query_calls - before_calls)
            if state.fallback_truncated:
                stats.inc("retriever.graph_fallback_truncated")

        for fact in candidates:
            if _add_fact(fact, facts, seen_ids):
                found.append(fact)
                if len(facts) >= _MAX_GRAPH_FACTS:
                    break
        return found

    def _expand_graph_two_hop(self, seed_entities: list[str]) -> list[Fact]:
        facts: list[Fact] = []
        seen_ids: set[str] = set()
        seen_entities: set[str] = set()
        state = _GraphExpansionState()
        frontier = _bounded_unique(seed_entities, limit=_MAX_GRAPH_FRONTIER)

        for _hop in range(2):
            variants = _entity_variants_for_frontier(frontier, seen_entities, limit=_MAX_GRAPH_FRONTIER)
            if not variants:
                break
            found = self._query_graph_entities(variants, facts, seen_ids, state)
            if not found or len(facts) >= _MAX_GRAPH_FACTS:
                break
            if state.fallback_truncated:
                break
            frontier = _next_graph_frontier(found, seen_entities, limit=_MAX_GRAPH_FRONTIER)
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
        stats.inc("retriever.retrieve_calls")
        profile = self._planner.profile(intent, mode=mode)
        rerank_budget = reranker_candidate_budget(mode or intent)
        priority_scores = _intent_type_priority_scores(profile)
        raw_ents = self._extractor.extract(query)
        linked_ents = self._linker.link_all(raw_ents)
        ents = _compound_entities(linked_ents)
        scope_filter = RetrievalScopeFilter.from_raw(scope)
        domain_weights = _domain_weights(query, self._domain_graph, scope_filter)

        semantic_chunks: list[MemoryObject] = []
        if profile.semantic and _memory_type_allowed("note", scope_filter):
            stop_vec = stats.timeit("retriever.vec_ms")
            semantic_chunks = self._rank_memory_items(
                query,
                self._vector.semantic_search(query, k=max(k_sem * 3, k_sem)),
                domain_weights,
                scope_filter,
                memory_type="note",
                limit=k_sem,
                type_priority=priority_scores.get("note", 0.0),
                rerank_budget=rerank_budget,
            )
            stop_vec()

        facts: List[Fact] = []
        if profile.facts and _memory_type_allowed("fact", scope_filter):
            stop_kg = stats.timeit("retriever.kg_ms")
            facts = self._rank_memory_items(
                query,
                self._expand_graph_two_hop(ents),
                domain_weights,
                scope_filter,
                memory_type="fact",
                type_priority=priority_scores.get("fact", 0.0),
                rerank_budget=rerank_budget,
            )
            stop_kg()

        episodes: List[Episode] = []
        if profile.episodes and _memory_type_allowed("episode", scope_filter):
            stop_ep = stats.timeit("retriever.ep_ms")
            episodes = self._rank_memory_items(
                query,
                self._episodic.search(user=None, entities=ents, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="episode",
                limit=k_eps,
                type_priority=priority_scores.get("episode", 0.0),
                rerank_budget=rerank_budget,
            )
            stop_ep()

        decisions: List[DecisionRecord] = []
        if profile.decisions and self._decisions is not None and _memory_type_allowed("decision", scope_filter):
            decisions = self._rank_memory_items(
                query,
                self._decisions.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="decision",
                limit=k_eps,
                type_priority=priority_scores.get("decision", 0.0),
                rerank_budget=rerank_budget,
            )

        experiences: List[ExperienceRecord] = []
        if profile.experiences and self._experiences is not None and _memory_type_allowed("experience", scope_filter):
            experiences = self._rank_memory_items(
                query,
                self._experiences.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="experience",
                limit=k_eps,
                type_priority=priority_scores.get("experience", 0.0),
                rerank_budget=rerank_budget,
            )

        skills: List[SkillRecord] = []
        if profile.skills and self._skills is not None and _memory_type_allowed("skill", scope_filter):
            skills = self._rank_memory_items(
                query,
                self._skills.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="skill",
                limit=k_eps,
                type_priority=priority_scores.get("skill", 0.0),
                rerank_budget=rerank_budget,
            )

        patterns: List[PatternRecord] = []
        if profile.failure_patterns and self._patterns is not None and _memory_type_allowed("failure_pattern", scope_filter):
            patterns = self._rank_memory_items(
                query,
                self._patterns.search(query, k=max(k_eps * 3, k_eps)),
                domain_weights,
                scope_filter,
                memory_type="failure_pattern",
                limit=k_eps,
                type_priority=priority_scores.get("failure_pattern", 0.0),
                rerank_budget=rerank_budget,
            )

        facts = _deduplicate_items(facts, memory_type="fact")
        beliefs = build_fact_beliefs(facts) if profile.beliefs else []
        bundle = RetrievalBundle(
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
        return _deduplicate_retrieval_bundle(bundle, profile)

    def _rank_memory_items(
        self,
        query: str,
        items: list[Any],
        domain_weights: dict[str, float],
        scope_filter: RetrievalScopeFilter,
        *,
        memory_type: str,
        limit: int | None = None,
        type_priority: float = 0.0,
        rerank_budget: int | None = None,
    ) -> list[Any]:
        stop_rank = stats.timeit(f"retriever.rank.{memory_type}_ms")
        fingerprint_cache: dict[tuple[str, int], str] = {}
        scored: list[tuple[tuple[float, float, float, int], Any]] = []
        for idx, item in enumerate(items):
            if not _passes_scope_filter(item, memory_type=memory_type, scope_filter=scope_filter):
                continue
            score = _memory_score(item, domain_weights, scope_filter=scope_filter)
            if score < _retrieval_score_threshold(memory_type):
                continue
            scored.append(((score + type_priority, score, _memory_time(item), -idx), item))
        stats.inc(f"retriever.rank.{memory_type}.input", len(items))
        stats.inc(f"retriever.rank.{memory_type}.scored", len(scored))

        if not scored:
            stop_rank()
            return []

        deduped = _deduplicate_scored_items(scored, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
        stats.inc(f"retriever.rank.{memory_type}.dedup_dropped", len(scored) - len(deduped))
        pre_ranked = _top_scored_items(deduped, limit=_rerank_pool_size(limit, budget=rerank_budget))
        reranked = _deduplicate_items(self._rerank(query, pre_ranked), memory_type=memory_type, fingerprint_cache=fingerprint_cache)
        if limit is not None and len(reranked) < limit:
            reranked = _fill_from_pre_ranked(
                reranked,
                _top_scored_items(deduped, limit=limit),
                memory_type=memory_type,
                fingerprint_cache=fingerprint_cache,
            )
        stop_rank()
        if limit is not None:
            return reranked[: max(limit, 0)]
        return reranked

    def _rerank(self, query: str, items: list[Any]) -> list[Any]:
        if self._reranker is None or len(items) <= 1:
            return items
        stats.inc("retriever.reranker_calls")
        stats.inc("retriever.reranker_items", len(items))
        stop_rerank = stats.timeit("retriever.reranker_ms")
        with telemetry_span("retriever.rerank", item_count=len(items), reranker=type(self._reranker).__name__) as span:
            try:
                reranked = list(self._reranker.rerank(query, items))
            except Exception:
                stats.inc("retriever.reranker_fallback_count")
                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute("fallback", True)
                stop_rerank()
                return items
            stop_rerank()
            valid = _valid_reranked_items(reranked, items)
            if span is not None and hasattr(span, "set_attribute"):
                span.set_attribute("fallback", False)
                span.set_attribute("result_count", len(valid))
            return valid


def _deduplicate_scored_items(
    scored: list[tuple[tuple[float, ...], Any]],
    *,
    memory_type: str,
    fingerprint_cache: dict[tuple[str, int], str] | None = None,
) -> list[tuple[tuple[float, ...], Any]]:
    out: list[tuple[tuple[float, ...], Any]] = []
    seen: set[str] = set()
    for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True):
        keys = _dedup_keys(item, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
        if keys and seen.intersection(keys):
            continue
        seen.update(keys)
        out.append((score, item))
    return out


def _deduplicate_items(
    items: list[Any],
    *,
    memory_type: str,
    fingerprint_cache: dict[tuple[str, int], str] | None = None,
) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in items:
        keys = _dedup_keys(item, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
        if keys and seen.intersection(keys):
            continue
        seen.update(keys)
        out.append(item)
    return out


def _deduplicate_retrieval_bundle(bundle: RetrievalBundle, profile: Any) -> RetrievalBundle:
    seen_content: set[str] = set()
    fingerprint_cache: dict[tuple[str, int], str] = {}
    cleaned: dict[str, list[Any]] = {attr: [] for attr, _memory_type in _DEFAULT_BUNDLE_ORDER}

    for attr, memory_type in _bundle_section_order(profile):
        items = getattr(bundle, attr)
        for item in items:
            fingerprint = _content_fingerprint_cached(item, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
            if fingerprint and fingerprint in seen_content:
                continue
            if fingerprint:
                seen_content.add(fingerprint)
            cleaned[attr].append(item)

    return RetrievalBundle(
        semantic_chunks=cleaned["semantic_chunks"],
        facts=cleaned["facts"],
        beliefs=bundle.beliefs,
        episodes=cleaned["episodes"],
        decisions=cleaned["decisions"],
        experiences=cleaned["experiences"],
        skills=cleaned["skills"],
        failure_patterns=cleaned["failure_patterns"],
        citations=bundle.citations,
    )


def _bundle_section_order(profile: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen_attrs: set[str] = set()
    for section in getattr(profile, "context_sections", ()):
        pair = _BUNDLE_SECTION_TYPES.get(section)
        if pair is None:
            continue
        attr, _memory_type = pair
        if attr in seen_attrs:
            continue
        seen_attrs.add(attr)
        out.append(pair)
    for pair in _DEFAULT_BUNDLE_ORDER:
        attr, _memory_type = pair
        if attr not in seen_attrs:
            seen_attrs.add(attr)
            out.append(pair)
    return out


def _intent_type_priority_scores(profile: Any) -> dict[str, float]:
    ordered_types: list[str] = []
    seen: set[str] = set()
    for _attr, memory_type in _bundle_section_order(profile):
        normalized = _normalize_memory_type(memory_type)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered_types.append(normalized)
    total = len(ordered_types)
    return {memory_type: (total - idx) * _INTENT_PRIORITY_STEP for idx, memory_type in enumerate(ordered_types)}


def _dedup_keys(
    item: Any,
    *,
    memory_type: str,
    fingerprint_cache: dict[tuple[str, int], str] | None = None,
) -> set[str]:
    keys: set[str] = set()
    item_id = str(getattr(item, "id", "") or "").strip()
    if item_id:
        keys.add(f"{memory_type}:id:{item_id}")
    fingerprint = _content_fingerprint_cached(item, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
    if fingerprint:
        keys.add(f"{memory_type}:content:{fingerprint}")
    return keys


def _content_fingerprint_cached(
    item: Any,
    *,
    memory_type: str,
    fingerprint_cache: dict[tuple[str, int], str] | None,
) -> str:
    if fingerprint_cache is None:
        return _content_fingerprint(item, memory_type=memory_type)
    key = (memory_type, id(item))
    if key not in fingerprint_cache:
        fingerprint_cache[key] = _content_fingerprint(item, memory_type=memory_type)
        stats.inc("retriever.fingerprint_cache_misses")
    else:
        stats.inc("retriever.fingerprint_cache_hits")
    return fingerprint_cache[key]


def _content_fingerprint(item: Any, *, memory_type: str) -> str:
    if isinstance(item, MemoryObject):
        payload = item.payload or {}
        text = payload.get("text") or payload.get("content") or str(payload)
    elif isinstance(item, Fact):
        text = f"{item.subject} {item.predicate} {item.object}"
    elif isinstance(item, Episode):
        text = " ".join([item.summary, *[event.summary for event in item.events]])
    elif isinstance(item, DecisionRecord):
        text = " ".join([item.title, item.context, item.decision, " ".join(item.consequences)])
    elif isinstance(item, ExperienceRecord):
        text = " ".join([item.goal, item.context, item.decision, " ".join(item.actions), item.outcome, item.lesson, " ".join(item.reuse_when)])
    elif isinstance(item, SkillRecord):
        text = " ".join([item.name, item.problem, " ".join(item.procedure), " ".join(item.reuse_when)])
    elif isinstance(item, PatternRecord):
        text = " ".join([item.symptom, item.root_cause, item.fix, item.detection])
    else:
        text = str(item)
    return _normalize_text(text)


def _top_scored_items(scored: list[tuple[tuple[float, ...], Any]], *, limit: int | None) -> list[Any]:
    if limit is not None:
        if limit <= 0:
            return []
        pairs = heapq.nlargest(limit, scored, key=lambda pair: pair[0])
    else:
        pairs = sorted(scored, key=lambda pair: pair[0], reverse=True)
    return [item for _, item in pairs]


def _fill_from_pre_ranked(
    reranked: list[Any],
    pre_ranked: list[Any],
    *,
    memory_type: str,
    fingerprint_cache: dict[tuple[str, int], str] | None = None,
) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for item in reranked:
        keys = _dedup_keys(item, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
        seen.update(keys)
        out.append(item)
    for item in pre_ranked:
        keys = _dedup_keys(item, memory_type=memory_type, fingerprint_cache=fingerprint_cache)
        if keys and seen.intersection(keys):
            continue
        seen.update(keys)
        out.append(item)
    return out


def _rerank_pool_size(limit: int | None, *, budget: int | None = None) -> int | None:
    if limit is None:
        return budget
    pool = max(limit, min(_RERANK_POOL_MIN, max(limit, 1) * _RERANK_POOL_MULTIPLIER))
    if budget is None:
        return pool
    return min(pool, max(0, budget))


def _valid_reranked_items(reranked: list[Any], original: list[Any]) -> list[Any]:
    original_by_identity = {id(item): item for item in original}
    seen: set[int] = set()
    out: list[Any] = []
    for item in reranked:
        original_item = original_by_identity.get(id(item))
        if original_item is None:
            continue
        identity = id(original_item)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(original_item)
    for item in original:
        identity = id(item)
        if identity not in seen:
            out.append(item)
    return out


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


def _memory_score(item: Any, domain_weights: dict[str, float], *, scope_filter: RetrievalScopeFilter) -> float:
    score = 0.0
    scope = _scope(item)
    item_domain_ids = _item_domain_ids(item)
    for domain_id in item_domain_ids:
        score += domain_weights.get(domain_id, 0.0)

    environment = str(scope.get("environment") or "").lower()
    durability = str(scope.get("durability") or "").lower()
    excluded = bool(scope.get("exclude_from_consolidation") or _meta(item).get("exclude_from_consolidation"))

    if (domain_weights or scope_filter.domain_ids) and durability == "durable" and not item_domain_ids:
        score -= _UNSCOPED_DURABLE_PENALTY
    if environment in {"test", "benchmark", "sandbox"}:
        score -= 2.0
    if durability in {"ephemeral", "session"}:
        score -= 2.0
    if excluded:
        score -= 1.0
    return score


def _retrieval_score_threshold(memory_type: str) -> float:
    return float("-inf")


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
        item_domains = set(_item_domain_ids(item))
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


def _item_domain_ids(item: Any) -> list[str]:
    meta = _meta(item)
    raw: list[Any] = []
    raw.extend(_scope_domain_ids(_scope(item)))
    for key in ("domain_ids", "domains"):
        value = meta.get(key)
        if isinstance(value, str):
            raw.append(value)
        elif value:
            raw.extend(list(value))
    if meta.get("domain"):
        raw.append(str(meta["domain"]))
    return _bounded_unique([str(value) for value in raw], limit=32)


def _scope_domain_ids(scope: dict[str, Any]) -> list[str]:
    raw = scope.get("domain_ids") or []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if str(item)]


def _bounded_unique(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _entity_variants_for_frontier(frontier: list[str], seen_entities: set[str], *, limit: int) -> list[str]:
    variants: list[str] = []
    for entity in frontier:
        for variant in _entity_variants(entity):
            key = variant.casefold()
            if key in seen_entities:
                continue
            seen_entities.add(key)
            variants.append(variant)
            if len(variants) >= limit:
                return variants
    return variants


def _next_graph_frontier(facts: list[Fact], seen_entities: set[str], *, limit: int) -> list[str]:
    frontier: list[str] = []
    for fact in facts:
        for value in (fact.subject, fact.object):
            key = str(value or "").casefold()
            if not value or key in seen_entities:
                continue
            frontier.append(str(value))
            if len(frontier) >= limit:
                return frontier
    return frontier


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

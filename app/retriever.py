# app/retriever.py
from __future__ import annotations

from typing import List, Set

from domain.models import RetrievalBundle, Fact, Episode
from domain.ports import IRetriever, IMemoryReadRepository, IGraphRepository, IEpisodicRepository
from app.entities import build_entity_stack
from app.config import settings
from app.profiling import timed
from app.stats import stats

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


class Retriever(IRetriever):
    """
    Объединённый извлекатель:
      - semantic: векторный поиск по исходному запросу
      - graph: выборка фактов по найденным сущностям (subject/object)
      - episodic: поиск эпизодов по сущностям
    """

    def __init__(
        self,
        vector_repo: IMemoryReadRepository,
        graph_repo: IGraphRepository,
        episodic_repo: IEpisodicRepository,
    ) -> None:
        self._vector = vector_repo
        self._graph = graph_repo
        self._episodic = episodic_repo
        self._extractor, self._linker = build_entity_stack(settings.ner_backend, settings.entity_aliases)

    @timed("retriever.retrieve", slow_ms=100)
    def retrieve(self, query: str, k_sem: int = 5, k_eps: int = 3) -> RetrievalBundle:
        raw_ents = self._extractor.extract(query)
        ents = self._linker.link_all(raw_ents)

        # I Семантические чанки
        stop_vec = stats.timeit("retriever.vec_ms")
        semantic_chunks = self._vector.semantic_search(query, k=k_sem)
        stop_vec()
        
        
        # II Факты (по subject и по object для каждой выделенной сущности)
        stop_kg = stats.timeit("retriever.kg_ms")
        facts: List[Fact] = []
        seen_ids: Set[str] = set()
        for e in ents:
            # query по subject
            for f in self._graph.query(subject=e):
                if f.id not in seen_ids:
                    seen_ids.add(f.id); facts.append(f)
            # query по object
            for f in self._graph.query(object=e):
                if f.id not in seen_ids:
                    seen_ids.add(f.id); facts.append(f)
        stop_kg()
        
        # III Эпизоды (пользователя пока не извлекаем -> None)
        stop_ep = stats.timeit("retriever.ep_ms")
        episodes: List[Episode] = self._episodic.search(user=None, entities=ents, k=k_eps)
        stop_ep()
        
        return RetrievalBundle(semantic_chunks=semantic_chunks, facts=facts, episodes=episodes, citations=[])

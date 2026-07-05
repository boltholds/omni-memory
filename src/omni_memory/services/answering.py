# app/services/answering.py (или там, где собираете ответ)
import re
import time
from omni_memory.metrics import QA_HALLUS, QA_CONFLICT_MISS, QA_ANSWERS, QA_CONSIST, QA_JUDGE_LAT

HEDGE_RE = re.compile(r"\b(don't know|unknown|insufficient|no (?:data|context))\b", re.I)

def score_consistency(text: str, used_sections: list[dict]) -> float:
    """Грубая эвристика 0..1: есть ли цитаты/ключевые слова из used_sections внутри ответа."""
    if not used_sections:
        return 0.0
    bullets = []
    for s in used_sections:
        body = (s.get("body") or "")[:2000]
        bullets.extend([b.strip("- ").lower() for b in body.splitlines() if b.strip().startswith("-")])
    if not bullets:
        return 0.5  # нейтрально
    hits = sum(1 for b in bullets if b and b in text.lower())
    return min(1.0, hits / max(3, len(bullets)))

def quality_judge(answer_text: str, used_sections: list[dict], conflicts: list[dict]) -> None:
    t0 = time.perf_counter()
    try:
        # 1) hallucination: нет секций и нет хеджа
        if not used_sections and not HEDGE_RE.search(answer_text):
            QA_HALLUS.labels("no_sources").inc()

        # 2) conflict miss: есть конфликты по запросу, но ответ их не отражает
        has_conf = bool(conflicts)
        mentions_conf = "conflict" in answer_text.lower() or "противореч" in answer_text.lower()
        if has_conf and not mentions_conf:
            QA_CONFLICT_MISS.inc()

        # 3) consistency score
        QA_ANSWERS.inc()
        QA_CONSIST.observe(score_consistency(answer_text, used_sections))
    finally:
        QA_JUDGE_LAT.observe(time.perf_counter() - t0)

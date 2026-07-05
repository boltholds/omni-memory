# infra/distillers/rule_based.py

import re
from typing import Optional

from omni_memory.domain.distiller import DistillationResult, DistilledFact, IMemoryDistiller


class RuleBasedMemoryDistiller(IMemoryDistiller):
    def __init__(self, patterns: Optional[list[str]] = None):
        self.patterns = patterns if patterns else [
            (
                r"(?P<subject>\w+)\s+lives\s+in\s+(?:the\s+)?(?P<object>[\w\s-]+)",
                "location",
            ),
            (
                r"(?P<subject>\w+)\s+moved\s+to\s+(?:the\s+)?(?P<object>[\w\s-]+)",
                "location",
            ),
            (
                r"(?P<subject>\w+)\s+works\s+with\s+(?P<object>[\w\s-]+)",
                "works_with",
            ),
        ]
    
    
    def distill(self, text: str) -> DistillationResult:
        facts = []



        normalized = text.strip()

        for pattern, predicate in self.patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if not match:
                continue

            facts.append(
                DistilledFact(
                    subject=match.group("subject").strip(),
                    predicate=predicate,
                    object=match.group("object").strip(),
                    confidence=0.7,
                    evidence=match.group(0),
                    volatility="medium",
                )
            )

        if not facts:
            return DistillationResult(rejected=["No structured facts extracted."])

        return DistillationResult(facts=facts)
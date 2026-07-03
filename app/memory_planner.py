from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MEMORY_INTENT = "full"

@dataclass(frozen=True)
class MemoryIntentProfile:
    name: str
    semantic: bool = True
    facts: bool = True
    beliefs: bool = True
    conflicts: bool = True
    episodes: bool = True
    decisions: bool = True
    experiences: bool = True
    skills: bool = True
    failure_patterns: bool = True
    context_sections: tuple[str, ...] = (
        "conflicts",
        "current_beliefs",
        "facts",
        "episodes",
        "decisions",
        "relevant_experience",
        "skills",
        "failure_patterns",
        "semantic_notes",
    )

class MemoryPlanner:
    def __init__(self, profiles: dict[str, MemoryIntentProfile] | None = None) -> None:
        self._profiles = {profile.name: profile for profile in _default_profiles()}
        if profiles:
            self._profiles.update(profiles)

    def profile(self, intent: str | None = None, *, mode: str | None = None) -> MemoryIntentProfile:
        requested = _normalize_intent(intent or mode)
        return self._profiles.get(requested) or self._profiles[DEFAULT_MEMORY_INTENT]

def _default_profiles() -> list[MemoryIntentProfile]:
    return [
        MemoryIntentProfile(name="full"),
        MemoryIntentProfile(
            name="answer_question",
            semantic=True,
            facts=True,
            beliefs=True,
            conflicts=True,
            episodes=False,
            decisions=False,
            experiences=False,
            skills=False,
            failure_patterns=False,
            context_sections=("conflicts", "current_beliefs", "facts", "semantic_notes"),
        ),
        MemoryIntentProfile(
            name="make_decision",
            semantic=False,
            facts=False,
            beliefs=False,
            conflicts=False,
            episodes=False,
            decisions=True,
            experiences=True,
            skills=True,
            failure_patterns=False,
            context_sections=("decisions", "relevant_experience", "skills"),
        ),
        MemoryIntentProfile(
            name="debug_failure",
            semantic=False,
            facts=False,
            beliefs=False,
            conflicts=False,
            episodes=False,
            decisions=False,
            experiences=True,
            skills=True,
            failure_patterns=True,
            context_sections=("failure_patterns", "skills", "relevant_experience"),
        ),
        MemoryIntentProfile(
            name="plan_task",
            semantic=True,
            facts=True,
            beliefs=True,
            conflicts=False,
            episodes=False,
            decisions=True,
            experiences=True,
            skills=True,
            failure_patterns=True,
            context_sections=("current_beliefs", "facts", "decisions", "skills", "failure_patterns", "relevant_experience", "semantic_notes"),
        ),
        MemoryIntentProfile(
            name="write_code",
            semantic=True,
            facts=False,
            beliefs=False,
            conflicts=False,
            episodes=False,
            decisions=True,
            experiences=True,
            skills=True,
            failure_patterns=True,
            context_sections=("decisions", "skills", "failure_patterns", "relevant_experience", "semantic_notes"),
        ),
    ]

def _normalize_intent(value: str | None) -> str:
    normalized = str(value or DEFAULT_MEMORY_INTENT).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "answer": "answer_question",
        "qa": "answer_question",
        "question": "answer_question",
        "default": "full",
        "all": "full",
        "decision": "make_decision",
        "decide": "make_decision",
        "debug": "debug_failure",
        "failure": "debug_failure",
        "plan": "plan_task",
        "task": "plan_task",
        "code": "write_code",
        "coding": "write_code",
    }
    return aliases.get(normalized, normalized)

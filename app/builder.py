from __future__ import annotations

from app.memory import OmniMemory


def build_memory(
    *,
    use_llm: bool = False,
    reject_conflicts: bool = False,
) -> OmniMemory:
    """Build the central OmniMemory facade used by CLI, FastAPI and examples."""
    return OmniMemory(
        use_llm=use_llm,
        reject_conflicts=reject_conflicts,
    )

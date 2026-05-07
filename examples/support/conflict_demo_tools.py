from __future__ import annotations

from domain.models import ConflictReport, RetrievalBundle


def render_debug(bundle: RetrievalBundle, conflict_report: ConflictReport, answer: str) -> str:
    lines: list[str] = ["=== FACTS ==="]
    lines.extend(f"- {fact.subject} {fact.predicate} {fact.object}" for fact in bundle.facts)

    lines.append("\n=== CONFLICTS ===")
    lines.extend(
        f"- {conflict.key}: {', '.join(conflict.variants)}"
        for conflict in conflict_report.conflicts
    )

    lines.append("\n=== ANSWER ===")
    lines.append(answer)
    return "\n".join(lines)


def demo_conflict_answer(bundle: RetrievalBundle, conflict_report: ConflictReport) -> str:
    if not conflict_report.conflicts:
        return "No conflicts found."

    conflict = conflict_report.conflicts[0]
    selected = conflict.variants[-1]

    for fact in bundle.facts:
        if f"{fact.subject}::{fact.predicate}" == conflict.key:
            selected = fact.object

    return (
        f"{selected} is the most likely answer.\n\n"
        "However, conflicting memory was found:\n"
        f"- {', '.join(conflict.variants)}\n\n"
        f"Selected: {selected}."
    )

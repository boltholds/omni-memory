from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.agent_cycle import AgentCycleRecord


class OpsCycleDraft(BaseModel):
    """Operations/incident workflow adapter for AgentCycleRecord."""

    goal: str
    service: str
    alert_id: str | None = None
    symptoms: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    outcome: str = ""
    metrics_before: dict[str, float] = Field(default_factory=dict)
    metrics_after: dict[str, float] = Field(default_factory=dict)
    lesson: str = ""
    reuse_when: list[str] = Field(default_factory=list)
    avoid_when: list[str] = Field(default_factory=list)
    affected_resources: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_agent_cycle(self) -> AgentCycleRecord:
        resources = _resources(self)
        validation = {
            "service": self.service,
            "alert_id": self.alert_id,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "metric_deltas": _metric_deltas(self.metrics_before, self.metrics_after),
            "sla_restored": _sla_restored(self),
            "sla_missed": bool(self.meta.get("sla_missed", False)),
        }
        context = _context(self)
        refs = {
            "service": self.service,
            "alert_id": self.alert_id,
            "affected_resources": resources,
        }
        return AgentCycleRecord(
            goal=self.goal,
            context=context,
            plan=[],
            decisions=[],
            actions=self.actions,
            outcome=self.outcome,
            evaluation={"validation": validation},
            lesson=self.lesson or "Review this ops cycle and fill a reusable lesson before recording.",
            confidence=self.confidence,
            refs=refs,
            affected_resources=resources,
            validation=validation,
            domain="ops",
            meta={
                "recorded_from": "ops_cycle",
                "draft": not bool(self.lesson),
                "domain": "ops",
                "service": self.service,
                **self.meta,
            },
            reuse_when=self.reuse_when or self.symptoms,
            avoid_when=self.avoid_when,
        )


class OpsCycleRecorder:
    def draft(self, cycle: OpsCycleDraft | dict[str, Any]) -> AgentCycleRecord:
        parsed = cycle if isinstance(cycle, OpsCycleDraft) else OpsCycleDraft.model_validate(cycle)
        return parsed.to_agent_cycle()


def _resources(cycle: OpsCycleDraft) -> list[str]:
    resources = list(cycle.affected_resources)
    service_ref = f"service:{cycle.service}"
    if service_ref not in resources:
        resources.insert(0, service_ref)
    if cycle.alert_id:
        alert_ref = f"alert:{cycle.alert_id}"
        if alert_ref not in resources:
            resources.append(alert_ref)
    return resources


def _context(cycle: OpsCycleDraft) -> str:
    parts = [f"Service: {cycle.service}"]
    if cycle.alert_id:
        parts.append(f"Alert: {cycle.alert_id}")
    if cycle.symptoms:
        parts.append("Symptoms: " + " | ".join(cycle.symptoms))
    return "\n".join(parts)


def _metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, before_value in before.items():
        if key in after:
            out[key] = after[key] - before_value
    return out


def _sla_restored(cycle: OpsCycleDraft) -> bool:
    if "sla_restored" in cycle.meta:
        return bool(cycle.meta["sla_restored"])
    if not cycle.metrics_before or not cycle.metrics_after:
        return False
    improved = 0
    compared = 0
    for key, before_value in cycle.metrics_before.items():
        if key not in cycle.metrics_after:
            continue
        compared += 1
        after_value = cycle.metrics_after[key]
        if _lower_is_better(key):
            improved += int(after_value < before_value)
        else:
            improved += int(after_value > before_value)
    return compared > 0 and improved >= max(1, compared // 2)


def _lower_is_better(metric_name: str) -> bool:
    key = metric_name.casefold()
    return any(token in key for token in ["latency", "error", "failure", "fail", "cost", "duration", "queue", "saturation", "cpu", "memory"])

from __future__ import annotations

from typing import Any

from omni_memory.ops_cycle import OpsCycleDraft, OpsCycleRecorder
from omni_memory.domain.models import WriteReport


class OpsMemoryWorkflow:
    """Workflow facade for recording operations/incident cycles."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self.recorder = OpsCycleRecorder()

    def draft_cycle(self, cycle: OpsCycleDraft | dict[str, Any]):
        return self.recorder.draft(cycle)

    def record_cycle(self, cycle: OpsCycleDraft | dict[str, Any], *, source: str = "ops-workflow") -> WriteReport:
        draft = self.draft_cycle(cycle)
        return self.memory.record_agent_cycle(draft, source=source)

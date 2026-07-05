from __future__ import annotations

from omni_memory.builder import build_memory
from omni_memory.export_import import import_memory
from omni_memory.infra.embeddings.factory import HashEmbedder


def _memory():
    return build_memory(use_llm=False, embedder=HashEmbedder())


class MemoryImportFacade:
    def __init__(self, memory):
        self.memory = memory

    def write(self, items):
        return self.memory.write_items(items, source="import-test")


def test_import_memory_uses_writeback_facade_without_legacy_service():
    memory = _memory()
    archive = {
        "facts": [
            {
                "id": "fact-imported",
                "type": "fact",
                "subject": "import",
                "predicate": "uses",
                "object": "writeback service",
                "meta": {"confidence": 1.0, "domain_ids": ["domain:project:omni-memory"]},
            }
        ],
        "notes": [
            {"id": "note-imported", "payload": {"text": "Imported note through current writeback facade."}}
        ],
        "episodes": [],
    }

    report = import_memory(MemoryImportFacade(memory), archive)

    assert report.saved == 2
    assert report.rejected == 0
    assert memory.repository_stats()["facts"] == 1
    assert memory.repository_stats()["vector_objects"] == 1

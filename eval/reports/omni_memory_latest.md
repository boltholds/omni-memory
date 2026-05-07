# Evaluation report: omni_memory_no_llm

Cases: 9
Score: 72/72 (100.0%)

| Case | Total | Correct | Recall | Conflict | Hallucination | Notes |
|---|---:|---:|---:|---:|---:|---|
| dev_001_packaging_context | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: Project omni-memory uses Python 3.12, Poetry, pytest, package name omni_memory, and packages app/domain/infra.. |
| dev_002_cli_failure_context | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: CLI smoke test test_load_facts_and_notes failed with ConnectError WinError 10061 connection refused when load-facts tried to call a running service.. |
| dev_003_quality_eval_context | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: tests/test_quality_eval.py failed around hallucination_when_no_context in Starlette TestClient/httpx/anyio stack.. |
| conflict_001_location | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant facts: alice location bridge; alice location lighthouse. Conflicting memory was found: alice::location: bridge, lighthouse. |
| conflict_002_project_status | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant facts: omni-memory status prototype; omni-memory status production-ready. Conflicting memory was found: omni-memory::status: production-ready, prototype. |
| conflict_003_no_conflict | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant facts: bob works_with alice. |
| pref_001_short_answers | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: The user prefers short, practical answers with minimal theory.. |
| pref_002_poetry_pytest | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: The user prefers Poetry for Python packaging and pytest for tests.. |
| pref_003_russian_language | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: The user usually discusses this project in Russian.. |

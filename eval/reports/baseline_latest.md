# Evaluation report: baseline_keyword_rag

Cases: 9
Score: 62/72 (86.11%)

| Case | Total | Correct | Recall | Conflict | Hallucination | Notes |
|---|---:|---:|---:|---:|---:|---|
| dev_001_packaging_context | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: Project omni-memory uses Python 3.12, Poetry, pytest, package name omni_memory, and packages app/domain/infra. |
| dev_002_cli_failure_context | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: CLI smoke test test_load_facts_and_notes failed with ConnectError WinError 10061 connection refused when load-facts tried to call a running service. |
| dev_003_quality_eval_context | 4/8 | 0 | 0 | 2 | 2 | missing: test_quality_eval.py, hallucination_when_no_context |
| ↳ answer |  |  |  |  |  | I do not have enough memory context to answer. |
| conflict_001_location | 5/8 | 1 | 2 | 0 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant facts: alice location lighthouse; alice location bridge |
| conflict_002_project_status | 5/8 | 1 | 2 | 0 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant facts: omni-memory status prototype; omni-memory status production-ready |
| conflict_003_no_conflict | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant facts: bob works_with alice |
| pref_001_short_answers | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: The user prefers short, practical answers with minimal theory. |
| pref_002_poetry_pytest | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: The user prefers Poetry for Python packaging and pytest for tests. |
| pref_003_russian_language | 8/8 | 2 | 2 | 2 | 2 | ok |
| ↳ answer |  |  |  |  |  | Relevant notes: The user usually discusses this project in Russian. |

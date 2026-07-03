# OmniMemory memory evaluation benchmark

This benchmark compares the same LLM in two modes:

1. `no_memory`: the model receives only the current question.
2. `memory`: the model receives context assembled through OmniMemory after benchmark memories are written.

The default provider is a deterministic fake extractive LLM. It is intended for fast smoke tests of the memory layer and CI. For real model evaluation, pass an OpenAI-compatible or Ollama provider.

## Run smoke benchmark

```bash
python benchmarks/memory_eval/run_benchmark.py
```

Outputs:

```text
benchmark-results/memory_eval/results.jsonl
benchmark-results/memory_eval/summary.json
```

Render markdown report:

```bash
python benchmarks/memory_eval/report.py
```

Output:

```text
benchmark-results/memory_eval/report.md
```

## Run with a real local model

Example:

```bash
python benchmarks/memory_eval/run_benchmark.py \
  --provider openai-compatible \
  --base-url http://localhost:11434/v1 \
  --model llama3.1 \
  --temperature 0
```

## Case format

Each line in `cases/*.jsonl` is one benchmark case:

```json
{
  "id": "fact_update_001",
  "category": "fact_update",
  "lang": "en",
  "question": "What backend framework does project use now?",
  "setup_turns": [
    {"role": "user", "content": "Project backend framework was Flask."},
    {"role": "user", "content": "Project moved to FastAPI; Flask is no longer used."}
  ],
  "memory_items": [
    {"type": "fact", "subject": "project", "predicate": "backend_framework", "object": "FastAPI"}
  ],
  "expected_answer_contains": ["FastAPI"],
  "expected_answer_not_contains": ["Unknown"],
  "expected_context_contains": ["FastAPI"],
  "expected_saved_min": 1
}
```

`setup_turns` are the human-readable scenario. `memory_items` are the deterministic writeback inputs used by the current runner. This keeps the benchmark stable while still documenting the original dialogue that produced the memory.

`expected_context_contains` is optional. If it is omitted, the runner reuses `expected_answer_contains` to compute `context_score`. Use it when the context evidence and final answer wording should be scored differently.

Privacy-only cases such as `pii` are excluded from `context_score`, because the correct behavior is usually that sensitive evidence does not appear in the assembled context.

Future extension: add an `--ingest session` mode that uses session distillation from `setup_turns` instead of deterministic `memory_items`.

## Metrics

The runner computes:

- `no_memory_score`: answer score without OmniMemory context.
- `memory_score`: answer score with OmniMemory context and writeback checks.
- `context_score`: whether OmniMemory retrieved the expected evidence before answer generation.
- `memory_lift`: `memory_score - no_memory_score`.
- `privacy_violations`: cases where forbidden values appear in the answer or saved memory.
- `write_failures`: cases where expected writeback saved/rejected counts were not met.
- `context_failures`: cases where writeback may be fine, but retrieval/context assembly missed expected evidence.
- `answer_failures_with_context_ok`: cases where retrieval succeeded but final answer generation/scoring failed.

Each result row also includes:

- `policy_decisions`: conversion, write-policy and repository decisions produced during writeback.
- `memory_operations`: remember-operation envelopes with before/after payloads and final status.

The benchmark intentionally includes categories that measure different properties:

- `simple_fact`: basic factual recall.
- `fact_update`: stale fact handling and current belief selection.
- `conflict`: conflict-aware context and current belief selection.
- `conflict_reject`: writeback rejection on conflict.
- `preference`: personalization and semantic note/preference recall.
- `pii`: secrets and private data should not be saved or repeated.
- `noisy_session`: noisy or low-confidence memories should not dominate.
- `multihop`: early check for multi-hop memory retrieval limits.

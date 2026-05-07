# Omni-memory evaluation

Run the baseline:

```bash
poetry run python -m eval.runners.run_baseline_rag
```

Run omni-memory without external LLM calls:

```bash
poetry run python -m eval.runners.run_omni_memory
```

The useful comparison is not the absolute score. The useful comparison is whether `omni_memory_no_llm` beats `baseline_keyword_rag` on conflict awareness, memory recall, and hallucination avoidance.

Current datasets:

- `eval/datasets/developer_assistant_cases.jsonl`
- `eval/datasets/conflict_cases.jsonl`
- `eval/datasets/preference_cases.jsonl`

Scoring scale per case: 8 points total.

- answer correctness: 0–2
- memory recall: 0–2
- conflict awareness: 0–2
- hallucination avoidance: 0–2

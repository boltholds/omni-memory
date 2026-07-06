# Dev-Agent Memory Utility Benchmark

This benchmark measures whether memory helps a coding agent avoid repeated project-specific failures.

It compares three modes on the same cases and model:

```text
no_memory    task only
rag_only     task + untyped text note
omni_memory  task + intent-aware OmniMemory context with decisions/experience/failure patterns
```

The MVP uses a structured action contract and optional patch-level fixtures instead of editing the real repository. Each case defines:

```text
task
known_failure
seed memory record
expected action terms
forbidden repeated-mistake terms
```

Some cases also define tiny fixture files plus `patches` checks. In those cases the agent must return:

```json
{
  "patches": [
    {"path": "file.py", "find": "old text", "replace": "new text"}
  ]
}
```

The runner applies the patch in memory and verifies the fixture checks. This gives a second-level signal: not only "picked the right action", but "produced an edit that would pass the case test".

Some cases go one step further and define a sandbox mini-repository with pytest tests. Enable that layer with:

```bash
poetry run python benchmarks/dev_agent/run.py --provider fake --repo-eval
```

With `--repo-eval`, the runner creates a temporary repo, writes the fixture files, applies the agent patches, runs the case test command, and records whether pytest passed. The real project checkout is never modified.

The main metric is repeat-failure reduction:

```text
repeat_failure_reduction_vs_no_memory
repeat_failure_reduction_vs_rag_only
```

## Smoke Run

```bash
poetry run python benchmarks/dev_agent/run.py --provider fake
```

## Local Ollama Run

```bash
poetry run python benchmarks/dev_agent/run.py \
  --provider ollama \
  --model qwen2.5:7b-instruct \
  --timeout 120 \
  --modes no_memory rag_only omni_memory \
  --repo-eval
```

For the OpenAI-compatible Ollama endpoint:

```bash
poetry run python benchmarks/dev_agent/run.py \
  --provider openai-compatible \
  --base-url http://localhost:11434/v1 \
  --model qwen2.5:7b-instruct \
  --api-key local
```

Results are written to:

```text
benchmark-results/dev_agent/results.jsonl
benchmark-results/dev_agent/summary.json
```

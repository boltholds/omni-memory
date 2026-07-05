# omni-memory MVP

LLM agents forget, duplicate facts, and hallucinate over outdated memory. Omni Memory gives agents structured long-term memory with conflict detection, write-back policies, current-belief resolution, and explainable context.

## Start

Install dependencies and run a local readiness check:

```bash
poetry install
poetry run omni-memory doctor
```

Run the FastAPI server:

```bash
poetry run omni-memory serve --host 127.0.0.1 --port 8000
```

Run the MCP stdio server for MCP clients:

```bash
poetry run omni-memory mcp
```

The product CLI keeps the top-level command list short. Maintenance commands are grouped:

```bash
poetry run omni-memory memory write-note "OmniMemory stores governed memory."
poetry run omni-memory memory retrieve "governed memory"
poetry run omni-memory memory path
poetry run omni-memory admin export memory.json
poetry run omni-memory debug llm-check
```

The older short CLI name is still available as a full legacy command surface:

```bash
poetry run omem memory-path
```

### MCP client config

For clients that accept a command/args MCP config, use:

```json
{
  "mcpServers": {
    "omni-memory": {
      "command": "poetry",
      "args": ["run", "omni-memory", "mcp"]
    }
  }
}
```

After packaging/PyPI install, the command can be shortened to:

```json
{
  "mcpServers": {
    "omni-memory": {
      "command": "omni-memory",
      "args": ["mcp"]
    }
  }
}
```

## Agent framework quickstart

### LangChain tools

Install the optional integration group:

```bash
poetry install --with langchain
```

Connect OmniMemory to a LangChain-style agent in three lines:

```python
from omni_memory import build_memory
from omni_memory.integrations.langchain import create_omni_memory_tools

tools = create_omni_memory_tools(build_memory())
```

The tool bundle includes:

```text
omni_memory_retrieve
omni_memory_context
omni_memory_write
omni_memory_finish_development_task
omni_memory_consolidate
```

### LangGraph nodes

Use OmniMemory as graph state nodes without adding LangGraph as a core dependency:

```python
from omni_memory import build_memory
from omni_memory.integrations.langgraph import make_context_node

context_node = make_context_node(build_memory())
```

The node reads `query`, `memory_intent` and optional `memory_scope` from state and writes a structured context pack back to `state["context"]`.

Runnable examples:

```text
examples/langchain_tools.py
examples/langgraph_agent_memory.py
```

## Docs

```text
docs/architecture.md          architecture and stable/experimental boundaries
docs/demo_end_to_end.md       full product demo scenario
docs/memory_persistence.md    optional SQL audit persistence setup
docs/release_checklist.md     pre-merge and release checklist
docs/stability.md             module stability policy
```

## Why OmniMemory is not just RAG

Classic RAG usually follows this flow:

```text
question -> vector search over documents -> top-k chunks -> LLM answer
```

OmniMemory is a memory layer for agents. It has both a write path and a read path:

```text
interaction/event -> writeback policies -> typed memories -> storage
question -> semantic + graph + episodic retrieval -> current beliefs/conflicts -> structured context -> LLM answer
```

The important differences are:

- **Writeback before retrieval.** New information is not blindly embedded. It passes conversion and write policies first.
- **Typed memory.** Facts, semantic notes, preferences and episodes can be stored and retrieved differently.
- **Policy-first lifecycle.** Provenance, TTL, PII blocking, conflict checks, confidence checks and deduplication run before persistence.
- **Current beliefs.** Multiple historical facts can exist, while the context builder can expose the current best belief and preserve alternatives.
- **Conflict visibility.** Conflicting facts are surfaced instead of being left as unrelated chunks.
- **Intent-aware retrieval.** The memory planner can retrieve different memory channels for answering, planning, decisions, debugging and coding.
- **Inspectable context.** The API can return both the final context and the retrieval bundle used to build it.

This makes OmniMemory closer to a transparent long-term memory subsystem than to a document search layer.

## v1 API

### Remember memory

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "source": "demo",
    "policy_mode": "permissive",
    "items": [
      {
        "type": "fact",
        "subject": "project",
        "predicate": "backend_framework",
        "object": "FastAPI",
        "meta": {"confidence": 1.0}
      }
    ]
  }'
```

The response includes saved/rejected/error items, plus `policy_decisions` and `operations` for inspection.

### Policy modes

`/v1/memories/remember` accepts `policy_mode`:

```text
permissive -> conflicting facts are saved with meta.conflict
strict     -> conflicting facts are rejected with fact_conflict
review     -> conflicting facts are rejected with requires_review for future human approval UI
```

Example strict conflict check:

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/remember \
  -H "Content-Type: application/json" \
  -d '{
    "source": "demo",
    "policy_mode": "strict",
    "items": [
      {
        "type": "fact",
        "subject": "project",
        "predicate": "backend_framework",
        "object": "Flask",
        "meta": {"confidence": 1.0}
      }
    ]
  }'
```

If `project.backend_framework = FastAPI` already exists, this rejects the incoming `Flask` fact with a `conflict` policy decision.

### Search memory

```bash
curl -X POST http://127.0.0.1:8000/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{"q": "What backend framework does project use?", "intent": "answer_question"}'
```

This returns the raw retrieval bundle: semantic chunks, facts, current beliefs, episodes, decisions, experiences and citations.

### Build context

```bash
curl -X POST http://127.0.0.1:8000/v1/context \
  -H "Content-Type: application/json" \
  -d '{"q": "What backend framework does project use?", "intent": "answer_question", "max_tokens": 1200}'
```

This returns the structured context pack and the retrieval bundle used to build it.

### Inspect persistent audit records

When SQL audit persistence is enabled:

```bash
curl http://127.0.0.1:8000/v1/memories
curl http://127.0.0.1:8000/v1/audit/operations
curl http://127.0.0.1:8000/v1/audit/decisions
curl http://127.0.0.1:8000/v1/audit/reviews
```

## How to inspect why a memory was written

`/v1/memories/remember` returns an audit trail for each writeback operation.

Look at `policy_decisions` to answer:

```text
Which conversion policy matched?
Which write policies accepted the memory?
Which policy rejected it, if any?
Was it skipped because of dry_run?
Was it saved by the repository router?
```

A successful fact write should look conceptually like this:

```json
{
  "policy_decisions": [
    {"stage": "conversion", "policy": "fact_writeback", "action": "accept"},
    {"stage": "write_policy", "policy": "provenance", "action": "accept"},
    {"stage": "write_policy", "policy": "ttl", "action": "accept"},
    {"stage": "write_policy", "policy": "pii", "action": "accept"},
    {"stage": "write_policy", "policy": "conflict", "action": "accept"},
    {"stage": "write_policy", "policy": "confidence", "action": "accept"},
    {"stage": "repository", "policy": "repository_router", "action": "save"}
  ]
}
```

Look at `operations` to answer:

```text
What was the raw input?
What memory object was produced?
Was the operation saved, rejected, accepted as dry-run, or errored?
What was the before/after diff envelope?
```

This is the beginning of product-level memory governance: every saved memory can explain why it exists.

## Benchmark

Run the memory-vs-no-memory benchmark:

```bash
poetry run python benchmarks/memory_eval/run_benchmark.py --provider openai-compatible --base-url http://localhost:11434/v1 --model gemma3:1b --temperature 0
```

Render a markdown report:

```bash
poetry run python benchmarks/memory_eval/report.py
```

The benchmark reports answer score, context score, write failures, privacy violations and answer failures where context was already correct.

# OmniMemory

OmniMemory is a governed long-term memory server for coding and autonomous agents.

It helps an agent remember what it changed, why it changed it, what failed, which decisions were made, and which lessons should be reused in the next task. The first target use case is simple: put OmniMemory next to a coding agent through MCP and give the agent a durable project memory.

## What you get in 5 minutes

After setup, your agent can:

```text
read prior project context before starting work
record completed development tasks
remember reusable lessons and failure patterns
propose decision/ADR candidates for review
retrieve previous fixes when a similar bug appears again
explain why a memory was written through policy/audit metadata
```

The main workflow is not “chat history search”. It is a governed agent-memory loop:

```text
start task -> retrieve context -> make changes -> finish task -> write experience -> review proposals -> reuse memory later
```

## Golden path: coding agent memory through MCP

### 1. Install and check the runtime

```bash
poetry install
poetry run omni-memory doctor
```

Expected result:

```text
OmniMemory doctor: ok
[ok] python: ...
[ok] fastapi: FastAPI import
[ok] uvicorn: Uvicorn import
[ok] mcp: MCP SDK import
[ok] memory: repositories={...}
```

### 2. Start the MCP server

```bash
poetry run omni-memory mcp
```

This runs OmniMemory as an MCP stdio server. Your MCP client will start this command for you after you add it to the client config.

### 3. Connect it to an MCP client

For Claude Desktop, Cursor, Codex-style clients, or any client that accepts command/args MCP config, use:

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

After package installation outside Poetry, the config can be shortened to:

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

### Recommended MCP surface

OmniMemory exposes many MCP tools, but a coding agent usually should start with the small `agent_core` profile:

```text
omni_memory_context
omni_memory_retrieve
omni_memory_finish_development_task
omni_memory_search_experiences
omni_memory_search_failure_patterns
```

Those are enough for the normal loop: read relevant memory, use prior lessons/failure patterns, and record the completed task.

Keep maintenance/admin tools available for humans or trusted workflows:

```text
review queue tools
facts CRUD tools
consolidation tools
clear/stats/session tools
```

The full tool list remains available for advanced clients, but exposing everything to a fully autonomous coding agent is usually unnecessary.

### 4. Give your coding agent a memory instruction

Paste this into your coding agent/system instructions:

```text
Use OmniMemory as project memory.

Before starting a non-trivial coding task, call omni_memory_context or omni_memory_retrieve with the current task description and intent="write_code" or intent="debug_failure".

During the task, do not write every intermediate thought. Only write durable project facts, decisions, lessons, and completed task summaries.

When the task is complete, call omni_memory_finish_development_task with:
- goal: what the task tried to accomplish
- summary: what changed
- changed_files: files touched
- commands_run: commands executed
- tests: tests/checks run and their result
- decisions: architecture/product choices made
- outcome: final result
- lesson: reusable lesson for future agents
- reuse_when: when this memory should be retrieved again
- avoid_when: when this lesson should not be applied

If OmniMemory returns decision candidates or review items, surface them to the user instead of silently accepting them.
```

### 5. Ask the agent to finish a development task

Example user prompt to your coding agent:

```text
Implement the fix, run the relevant tests, then record the completed task in OmniMemory using omni_memory_finish_development_task. If OmniMemory proposes ADR/decision candidates, show them to me for review.
```

The important MCP tool for this path is:

```text
omni_memory_finish_development_task
```

It records the task as reusable experience and can return decision candidates for review instead of silently turning everything into permanent decisions.

### 6. Reuse memory in the next task

On a later task, ask:

```text
Before changing anything, query OmniMemory for prior decisions, failure patterns, and lessons related to this task.
```

The agent should call:

```text
omni_memory_context
omni_memory_retrieve
omni_memory_search_experiences
omni_memory_search_skills
omni_memory_search_failure_patterns
```

That is the “magic moment”: the next run is no longer starting from an empty context window.

## Demo: failure pattern changes the next agent run

The main product demo is deliberately simple:

```text
1. Agent without memory adds an MCP handler but forgets the schema registry.
2. The advertised-tools contract test fails.
3. OmniMemory records the failure pattern:
   handler and schema registry must be updated together.
4. A later agent receives a similar task and retrieves the failure pattern.
5. The agent updates registry + handler + contract test.
6. The test passes.
```

This is the behavior OmniMemory optimizes for: not just recalling facts, but helping the next development cycle avoid a previously learned mistake.

## What the agent should remember

Good memory candidates:

```text
project decisions and ADR-like choices
completed development task summaries
files/modules affected by a change
commands and tests that validated a fix
lessons that should be reused
failure patterns and their fixes
stable project facts with provenance
```

Bad memory candidates:

```text
temporary chain-of-thought
raw secrets, tokens, passwords, private keys
large logs or generated files
unverified guesses
personal data not needed for the project
one-off observations with no future reuse value
```

OmniMemory is policy-first: writes pass through conversion, provenance, TTL, PII, conflict, confidence, deduplication, and repository routing policies before persistence.

## Review mode and governance loop

Agents should not silently promote every idea into durable memory. OmniMemory supports review-oriented workflows through MCP tools such as:

```text
omni_memory_list_review_items
omni_memory_get_review_item
omni_memory_accept_review_item
omni_memory_reject_review_item
omni_memory_supersede_review_item
```

A practical review loop:

```text
agent finishes task
OmniMemory records experience
OmniMemory proposes decision/skill/failure-pattern candidates
human reviews pending items
accepted items become durable cognitive memory
rejected items stay out of the reusable memory layer
```

For consolidation, use:

```text
omni_memory_consolidate_experiences
```

By default, review/dry-run flows are safer for early adoption than automatically promoting everything.

## Where memory is stored

In local mode, OmniMemory stores data under:

```text
.omni-memory/
```

This includes JSON-backed durable stores and local vector index files. The exact files depend on which memory channels you use, but the local directory is the thing to back up for a simple single-user setup.

Useful CLI command:

```bash
poetry run omni-memory memory path
```

To reset local memory during development:

```bash
rm -rf .omni-memory
```

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force .omni-memory
```

For server/audit persistence, see:

```text
docs/memory_persistence.md
```

## Product CLI

The product CLI keeps the top level small:

```bash
poetry run omni-memory --help
```

Main commands:

```text
serve    run FastAPI server
mcp      run MCP stdio server
doctor   check local runtime readiness
memory   local memory read/write commands
admin    import/export and vector maintenance
debug    diagnostics and profiling
```

Examples:

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

## HTTP server quickstart

Run the FastAPI server:

```bash
poetry run omni-memory serve --host 127.0.0.1 --port 8000
```

Check health:

```bash
curl http://127.0.0.1:8000/healthz
```

By default, keep the server local while experimenting. Do not expose it on a public interface until you have an explicit authentication/security story for your deployment.

## Python integration

### Basic package API

```python
from omni_memory import build_memory

memory = build_memory(use_llm=False)
report = memory.write_note("Use FastAPI for the local HTTP server.", source="demo")
context = memory.build_context("What framework does the server use?")
```

### LangChain tools

Install the optional integration group:

```bash
poetry install --with langchain
```

Connect OmniMemory to a LangChain-style agent:

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

The node reads `query`, `memory_intent`, and optional `memory_scope` from state and writes a structured context pack back to `state["context"]`.

Runnable examples:

```text
examples/langchain_tools.py
examples/langgraph_agent_memory.py
```

## v1 HTTP API

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

This returns the raw retrieval bundle: semantic chunks, facts, current beliefs, episodes, decisions, experiences, and citations.

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

```text
writeback before retrieval
policy-governed persistence
typed memories: facts, notes, decisions, experiences, skills, failure patterns
current-belief resolution
conflict visibility
intent-aware retrieval
inspectable context and audit metadata
```

This makes OmniMemory closer to a transparent long-term memory subsystem than to a document search layer.

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

## Docs

```text
docs/architecture.md          architecture and stable/experimental boundaries
docs/demo_end_to_end.md       full product demo scenario
docs/memory_persistence.md    optional SQL audit persistence setup
docs/release_checklist.md     pre-merge and release checklist
docs/stability.md             module stability policy
```

## Benchmark

Run the memory-vs-no-memory benchmark:

```bash
poetry run python benchmarks/memory_eval/run_benchmark.py --provider openai-compatible --base-url http://localhost:11434/v1 --model gemma3:1b --temperature 0
```

Render a markdown report:

```bash
poetry run python benchmarks/memory_eval/report.py
```

The benchmark reports answer score, context score, write failures, privacy violations, and answer failures where context was already correct.

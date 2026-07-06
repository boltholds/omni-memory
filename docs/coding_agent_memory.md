# Coding Agent Memory Loop

OmniMemory's first product wedge is coding-agent memory through MCP.

## Agent Instruction Template

Paste this into a coding agent that has OmniMemory MCP tools:

```text
Use OmniMemory as durable project memory.

Before a non-trivial coding task, call omni_memory_context or omni_memory_retrieve with the task and intent="write_code" or intent="debug_failure".

Use retrieved decisions, experiences, skills and failure patterns to avoid repeating old mistakes.

During the task, do not write every thought. Only record durable facts, decisions, lessons and completed task summaries.

At the end, call omni_memory_finish_development_task with:
- goal
- summary
- changed_files
- commands_run
- tests
- decisions
- outcome
- lesson
- reuse_when
- avoid_when

If OmniMemory returns decision candidates or review items, show them to the user. Do not silently accept generated governance memory.
```

## What To Remember

Good memory:

```text
completed task summaries
project decisions and ADR-like choices
commands and tests that validated a fix
files/modules affected by a change
lessons that should be reused
failure patterns and fixes
stable project facts with provenance
```

Bad memory:

```text
temporary chain-of-thought
secrets, tokens, passwords and private keys
large generated logs
unverified guesses
one-off observations with no future reuse
```

## Killer Demo

```text
1. Agent without memory adds an MCP handler but forgets the schema registry.
2. The advertised-tools contract test fails.
3. OmniMemory records the failure pattern.
4. A later agent receives a similar MCP task.
5. It retrieves the failure pattern.
6. It updates registry + handler + contract test.
7. The test passes.
```

Executable contract:

```bash
poetry run pytest -q tests/e2e/test_coding_agent_memory_loop.py
```

This demo checks the behavior that matters: memory changes the next development decision.

## Review Loop

When candidates are proposed:

```bash
poetry run omni-memory review list
poetry run omni-memory review get <item-id>
poetry run omni-memory review accept <item-id>
poetry run omni-memory review reject <item-id>
poetry run omni-memory review supersede <item-id> replacement.json
```

Review is the boundary between useful agent suggestions and durable project memory.

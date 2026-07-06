# MCP Quickstart

This is the recommended first path for OmniMemory: run it as a local MCP stdio server next to a coding agent.

## 1. Check the runtime

```bash
poetry install
poetry run omni-memory doctor
```

Expected readiness profile:

```text
runtime: ok
mcp: available
persistence: local .omni-memory
llm: not-configured
security: local-only-recommended
```

LLM support is optional for the memory loop. Retrieval, context, review and task recording work without an LLM.

## 2. Start the MCP server

```bash
poetry run omni-memory mcp
```

Most MCP clients start this command for you from their config.

## 3. Add client config

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

After package installation outside Poetry:

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

## 4. Recommended agent tools

For ordinary coding-agent use, expose the `agent_core` surface:

```text
omni_memory_context
omni_memory_retrieve
omni_memory_finish_development_task
omni_memory_search_experiences
omni_memory_search_failure_patterns
```

Keep maintenance tools for humans or trusted workflows:

```text
review queue tools
facts CRUD tools
consolidation tools
clear/stats/session tools
```

## 5. First task

Ask the agent:

```text
Before changing code, query OmniMemory for relevant decisions, experience and failure patterns. After the task is complete, call omni_memory_finish_development_task and show me any review candidates.
```

The magic moment is the next similar task: the agent should retrieve a prior lesson or failure pattern before deciding how to change the code.

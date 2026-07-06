# Local Persistence And Backup

Local mode is the default OmniMemory storage story for a single developer or one project agent.

## Source Of Truth

In local CLI/MCP mode, the source of truth is:

```text
.omni-memory/
  facts.json
  decisions.json
  experiences.json
  skills.json
  failure_patterns.json
  review_queue.json
  vector/
```

The JSON files hold durable structured memory. The `vector/` directory holds the local semantic index for notes/chunks.

Audit SQL persistence is separate and optional. See `docs/memory_persistence.md` for SQL audit tables.

## Inspect Storage

```bash
poetry run omni-memory memory path
poetry run omni-memory doctor
```

## Backup

Use the product CLI:

```bash
poetry run omni-memory admin backup omni-memory-backup.zip
```

Or back up the whole local memory directory manually:

```bash
tar -czf omni-memory-backup.tgz .omni-memory
```

PowerShell:

```powershell
Compress-Archive -Path .omni-memory -DestinationPath omni-memory-backup.zip
```

## Restore

Stop the MCP/HTTP process first, then restore `.omni-memory` into the project root:

```bash
poetry run omni-memory admin restore omni-memory-backup.zip
```

Use `--force` only when you intentionally want to replace the current local memory directory.

PowerShell:

```powershell
Expand-Archive -Path omni-memory-backup.zip -DestinationPath .
```

## Reset During Development

Only do this when you intentionally want to delete local memory:

```powershell
Remove-Item -Recurse -Force .omni-memory
```

For automation, prefer `omni_memory_clear` through MCP or admin maintenance workflows so the action is explicit and auditable.

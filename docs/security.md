# Security Notes

OmniMemory is local-first by default. Treat it as a project memory server that can contain sensitive development history.

## MCP

The recommended MCP setup runs over stdio. It is local to the client process and does not expose a network port.

Still, agents should not store:

```text
secrets
tokens
passwords
private keys
unnecessary personal data
```

The writeback policy includes PII/secret filtering, but callers should avoid sending unsafe material in the first place.

## HTTP Server

Run locally while experimenting:

```bash
poetry run omni-memory serve --host 127.0.0.1 --port 8000
```

Do not bind to `0.0.0.0` with the default admin key. The CLI warns when it detects that combination.

Set a real admin key before exposing admin endpoints:

```text
ADMIN_API_KEY=replace-this
```

Admin endpoints require:

```text
X-API-Key: replace-this
```

## Production Checklist

Before exposing HTTP outside localhost:

```text
set ADMIN_API_KEY
run behind TLS
restrict network access
decide retention and backup policy
review what agents are allowed to write
monitor logs and audit output
```

# omni-memory MVP

LLM agents forget, duplicate facts, and hallucinate over outdated memory. Omni Memory gives agents structured long-term memory with conflict detection, write-back, and explainable context.

## Start

```bash

poetry install


poetry run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
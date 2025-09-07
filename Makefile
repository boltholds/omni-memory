.PHONY: fmt lint test dev

fmt:
	poetry run ruff check . --fix

lint:
	poetry run ruff check .
	poetry run mypy .

test:
	poetry run pytest -q

dev:
	poetry run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

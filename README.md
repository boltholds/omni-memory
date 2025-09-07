# omni-memory MVP

Мини-система памяти для LLM: векторное хранилище, граф фактов, эпизодическое хранилище и оркестратор.

## Быстрый старт

```bash
# Установить зависимости
poetry install

# Запуск сервиса
poetry run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

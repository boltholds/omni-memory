# 1. Builder (поэзия ставит зависимости)
FROM python:3.12-slim AS builder
WORKDIR /app

ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Скопируем только метаданные и lock для быстрого кеша
COPY pyproject.toml poetry.lock* ./
# Включаем режим пакетирования
RUN poetry config virtualenvs.create false \
 && poetry install --only main --no-interaction --no-ansi

# 2. Runtime
FROM python:3.12-slim
WORKDIR /app

# Системные зависимости при необходимости (faiss-cpu колёса уже включены)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные либы и исходники
COPY --from=builder /usr/local /usr/local
COPY . .

# Папка для данных
RUN mkdir -p data

# Переменные окружения
ENV ENV=prod HOST=0.0.0.0 PORT=8000 LOG_LEVEL=info SQLITE_PATH=/app/data/omni.db

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

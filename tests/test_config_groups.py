from __future__ import annotations

from omni_memory.config import Settings


def test_settings_keep_flat_env_compatibility_and_expose_grouped_views(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_OLLAMA_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("MEMORY_DATABASE_URL", "postgresql://example/test")
    monkeypatch.setenv("NER_BACKEND", "regex")

    settings = Settings()

    assert settings.llm_provider == "ollama"
    assert settings.llm.provider == "ollama"
    assert settings.llm.ollama_model == "qwen2.5:7b-instruct"

    assert settings.embedding_backend == "hash"
    assert settings.embedding.backend == "hash"

    assert settings.memory_database_url == "postgresql://example/test"
    assert settings.memory.database_url == "postgresql://example/test"

    assert settings.ner_backend == "regex"
    assert settings.entity.ner_backend == "regex"


def test_settings_grouped_views_cover_operational_sections():
    settings = Settings()

    assert settings.app.name == settings.app_name
    assert settings.tokenizer.context_max_tokens == settings.context_max_tokens
    assert settings.distiller.provider == settings.distiller_provider
    assert settings.prompt.template_dir == settings.prompt_template_dir
    assert settings.logging.level == settings.log_level
    assert settings.trace.sample_rate == settings.trace_sample_rate
    assert settings.telemetry.enabled == settings.omni_telemetry_enabled
    assert settings.telemetry.exporter == settings.omni_otel_exporter
    assert settings.admin.request_id_header == settings.request_id_header
    assert settings.storage.sqlite_path == settings.sqlite_path

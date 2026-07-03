from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings


class AppSettings(BaseModel):
    name: str
    env: str
    host: str
    port: int


class EmbeddingSettings(BaseModel):
    backend: str
    model: str


class TokenizerSettings(BaseModel):
    backend: str
    model: str
    context_max_tokens: int


class LLMSettings(BaseModel):
    provider: str
    model: str
    ollama_model: str
    temperature: float
    openai_api_key: str
    openai_base_url: str
    ollama_base_url: str | None


class DistillerSettings(BaseModel):
    provider: str
    model: str | None
    api_key: str | None
    base_url: str | None
    temperature: float


class PromptSettings(BaseModel):
    template_dir: str
    system_template: str
    user_template: str
    default_lang: str
    default_style: str


class LoggingSettings(BaseModel):
    level: str
    json: bool
    file: str | None
    rotation_mb: int
    keep_files: int


class TraceSettings(BaseModel):
    sample_rate: float
    log_body: bool
    body_max: int
    redact: bool


class AdminSettings(BaseModel):
    api_key: str
    enabled: bool
    rate_limit_per_min: int
    rate_limit_burst: int
    request_id_header: str


class MemoryPersistenceSettings(BaseModel):
    database_url: str | None
    audit_enabled: bool
    audit_auto_create: bool
    audit_default_limit: int


class StorageSettings(BaseModel):
    sqlite_path: str


class EntitySettings(BaseModel):
    ner_backend: str
    aliases: Dict[str, List[str]]


class Settings(BaseSettings):
    """Flat env-compatible settings with grouped read-only views.

    Environment variables stay backward-compatible, e.g. `LLM_PROVIDER`,
    `EMBEDDING_BACKEND`, `MEMORY_DATABASE_URL`. Use grouped properties in new
    code for readability: `settings.llm.provider`, `settings.embedding.backend`,
    `settings.memory.database_url`.
    """

    app_name: str = "omni-memory"
    env: str = "dev"  # dev|prod
    host: str = "0.0.0.0"
    port: int = 8000

    embedding_backend: str = "auto"  # auto|st|hash
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    context_max_tokens: int = 8000
    tokenizer_backend: str = "auto"  # auto|tiktoken|simple
    tokenizer_model: str = "cl100k_base"

    llm_provider: str = "openai"  # none|openai|ollama|openai-compatible
    llm_model: str = "/home/vrai/models/Qwen2.5-7B-Instruct-AWQ"
    llm_ollama_model: str = "llama3.1"
    llm_temperature: float = 0.3
    openai_api_key: str = "EMPTY"
    openai_base_url: str = "http://10.22.0.6:11434/v1"
    ollama_base_url: str | None = None

    distiller_provider: str = "inherit"  # inherit|none|openai-compatible|openai|ollama
    distiller_model: str | None = None
    distiller_api_key: str | None = None
    distiller_base_url: str | None = None
    distiller_temperature: float = 0.0

    prompt_template_dir: str = "templates/prompt"
    prompt_system_template: str = "system.j2"
    prompt_user_template: str = "user.j2"
    default_lang: str = "en"
    default_style: str = "concise"

    log_level: str = "INFO"
    log_json: bool = True
    log_file: str | None = "logs/app.log"
    log_rotation_mb: int = 20
    log_keep_files: int = 5

    trace_sample_rate: float = 1.0
    trace_log_body: bool = False
    trace_body_max: int = 1000
    trace_redact: bool = True

    admin_api_key: str = "CHANGE_ME"
    enable_admin: bool = True
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 30
    request_id_header: str = "X-Request-Id"

    memory_database_url: str | None = None
    memory_audit_enabled: bool = False
    memory_audit_auto_create: bool = False
    memory_audit_default_limit: int = 50

    sqlite_path: str = ":memory:"
    ner_backend: str = "spacy"  # regex|spacy|auto
    entity_aliases: Dict[str, List[str]] = {
        "alice": ["alisa", "алиса"],
        "lighthouse": ["beacon", "phare", "маяк"],
        "bridge": ["pont", "puente", "мост"],
        "nikolai": ["nicholas", "николай"],
    }

    @property
    def app(self) -> AppSettings:
        return AppSettings(name=self.app_name, env=self.env, host=self.host, port=self.port)

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings(backend=self.embedding_backend, model=self.embedding_model)

    @property
    def tokenizer(self) -> TokenizerSettings:
        return TokenizerSettings(backend=self.tokenizer_backend, model=self.tokenizer_model, context_max_tokens=self.context_max_tokens)

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            provider=self.llm_provider,
            model=self.llm_model,
            ollama_model=self.llm_ollama_model,
            temperature=self.llm_temperature,
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url,
            ollama_base_url=self.ollama_base_url,
        )

    @property
    def distiller(self) -> DistillerSettings:
        return DistillerSettings(
            provider=self.distiller_provider,
            model=self.distiller_model,
            api_key=self.distiller_api_key,
            base_url=self.distiller_base_url,
            temperature=self.distiller_temperature,
        )

    @property
    def prompt(self) -> PromptSettings:
        return PromptSettings(
            template_dir=self.prompt_template_dir,
            system_template=self.prompt_system_template,
            user_template=self.prompt_user_template,
            default_lang=self.default_lang,
            default_style=self.default_style,
        )

    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings(level=self.log_level, json=self.log_json, file=self.log_file, rotation_mb=self.log_rotation_mb, keep_files=self.log_keep_files)

    @property
    def trace(self) -> TraceSettings:
        return TraceSettings(sample_rate=self.trace_sample_rate, log_body=self.trace_log_body, body_max=self.trace_body_max, redact=self.trace_redact)

    @property
    def admin(self) -> AdminSettings:
        return AdminSettings(
            api_key=self.admin_api_key,
            enabled=self.enable_admin,
            rate_limit_per_min=self.rate_limit_per_min,
            rate_limit_burst=self.rate_limit_burst,
            request_id_header=self.request_id_header,
        )

    @property
    def memory(self) -> MemoryPersistenceSettings:
        return MemoryPersistenceSettings(
            database_url=self.memory_database_url,
            audit_enabled=self.memory_audit_enabled,
            audit_auto_create=self.memory_audit_auto_create,
            audit_default_limit=self.memory_audit_default_limit,
        )

    @property
    def storage(self) -> StorageSettings:
        return StorageSettings(sqlite_path=self.sqlite_path)

    @property
    def entity(self) -> EntitySettings:
        return EntitySettings(ner_backend=self.ner_backend, aliases=self.entity_aliases)

    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

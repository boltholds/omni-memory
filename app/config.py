from pydantic_settings import BaseSettings
from typing import Dict, List

class Settings(BaseSettings):
    app_name: str = "omni-memory"
    env: str = "dev"                 # dev|prod
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    
    embedding_backend: str = "auto"   # auto|st|hash
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    context_max_tokens: int = 8000
    tokenizer_backend: str = "auto"   # auto|tiktoken|simple
    tokenizer_model: str = "cl100k_base"  # имя модели для tiktoken; можно "cl100k_base"
    llm_provider: str = "openai"           # none|openai|ollama (можно "auto" позже)
    llm_model: str = "/home/vrai/models/Qwen2.5-7B-Instruct-AWQ"       # для OpenAI
    llm_ollama_model: str = "llama3.1"   # для Ollama
    llm_temperature: float = 0.3
    openai_api_key: str  = "EMPTY"
    openai_base_url: str = "http://10.22.0.6:11434/v1"   # опционально: совместимые провайдеры
    ollama_base_url: str | None = None
    
    prompt_template_dir: str = "templates/prompt"
    prompt_system_template: str = "system.j2"
    prompt_user_template: str = "user.j2"
    # язык/стиль по умолчанию 
    default_lang: str = "en"           # "en" | "ru"
    default_style: str = "concise"     # "concise" | "bullets" | "detailed"
    
    log_level: str = "INFO"           # DEBUG|INFO|WARNING|ERROR
    log_json: bool = True             # JSON-формат логов
    log_file: str | None = "logs/app.log"
    log_rotation_mb: int = 20         # ротация ~20MB
    log_keep_files: int = 5

    trace_sample_rate: float = 1.0    # 0..1 — доля запросов для расширенного трейсинга
    trace_log_body: bool = False      # логировать тело запроса/ответа (осторожно)
    trace_body_max: int = 1000        # макс. символов тела
    trace_redact: bool = True         # включить маскировку PII

    # заголовки для корреляции
    header_request_id: str = "X-Request-ID"
    header_api_key: str = "X-API-Key"     # TODO: Create API_key politics
    
    # episodic storage
    sqlite_path: str = ":memory:"    # "data/omni.db" в prod
    ner_backend: str = "spacy"  # regex|spacy|auto
    entity_aliases: Dict[str, List[str]] = {
        "alice": ["alisa", "алиса"],
        "lighthouse": ["beacon", "phare", "маяк"],
        "bridge": ["pont", "puente", "мост"],
        "nikolai": ["nicholas", "николай"],
    }
    
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

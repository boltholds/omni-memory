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
    context_max_tokens: int = 400
    tokenizer_backend: str = "auto"   # auto|tiktoken|simple
    tokenizer_model: str = "cl100k_base"  # имя модели для tiktoken; можно "cl100k_base"
    # episodic storage
    sqlite_path: str = ":memory:"    # "data/omni.db" в prod
    ner_backend: str = "regex"  # regex|spacy|auto
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

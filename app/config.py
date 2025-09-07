from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "omni-memory"
    env: str = "dev"                 # dev|prod
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # episodic storage
    sqlite_path: str = ":memory:"    # "data/omni.db" в prod

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

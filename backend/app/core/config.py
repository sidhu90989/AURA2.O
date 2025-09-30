from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    app_name: str = "AURA2.O Backend"
    environment: str = "dev"
    debug: bool = True
    api_version: str = "v1"

    # External service keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    coqui_api_key: Optional[str] = None

    # Model config
    openai_model: str = "gpt-4o-mini"
    system_persona: str = (
        "You are AURA, an advanced, emotionally intelligent assistant. "
        "Be concise, context-aware, ethically aligned, and helpful."
    )
    provider_order: str = "openai"

    # Database and vector paths
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/aura"
    vector_store_dir: str = "./vector_store"

    # Memory config
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    memory_top_k: int = 5
    max_memory_tokens: int = 2048

    # Security
    allowed_origins: str = "*"
    auth_token_secret: str = "dev-secret-change"  # Replace in production

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not yet modeled
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore

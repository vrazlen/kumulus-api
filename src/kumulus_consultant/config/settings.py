# src/kumulus_consultant/config/settings.py
import logging
from typing import Literal, Optional
# CORRECTED: Use native Pydantic V1 BaseSettings
from pydantic import BaseSettings, SecretStr, validator, PostgresDsn

class Settings(BaseSettings):
    """
    Manages the application's configuration using Pydantic's V1 BaseSettings.
    """
    # --- Core Application Settings ---
    LLM_PROVIDER: Literal["OpenAI", "Ollama", "Google"] = "Google"
    ENVIRONMENT: Literal["DEV", "PROD"] = "DEV"
    LOG_LEVEL: int = logging.INFO
    
    # --- Model-Specific Settings ---
    OLLAMA_MODEL: str = "tinyllama"
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # --- API Keys & Secrets ---
    OPENAI_API_KEY: Optional[SecretStr] = None
    GEMINI_API_KEY: Optional[SecretStr] = None
    # LangChain's Google integration also looks for this variable name
    GOOGLE_API_KEY: Optional[SecretStr] = None

    # --- PostGIS Database Credentials ---
    POSTGRES_USER: str = "kumulus_user"
    POSTGRES_PASSWORD: SecretStr = "strong_password"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "kumulus_gis"
    
    # --- ML Model Configuration ---
    MODEL_PATH: str = "models/deeplabv3plus_resnet50_informal_settlements.pth"

    # --- RAG System Configuration ---
    VECTOR_DB_PATH: str = "data/processed/faiss_index"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    DATABASE_URL: Optional[PostgresDsn] = None

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+psycopg2",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD").get_secret_value(),
            host=values.get("POSTGRES_HOST"),
            port=str(values.get("POSTGRES_PORT")),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    # CORRECTED: Use the inner Config class for Pydantic V1
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Ensure GOOGLE_API_KEY is populated from GEMINI_API_KEY if not set directly
if settings.GEMINI_API_KEY and not settings.GOOGLE_API_KEY:
    settings.GOOGLE_API_KEY = settings.GEMINI_API_KEY
# src/kumulus_consultant/config/settings.py
import logging
from typing import Literal, Optional, Any
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Manages the application's configuration using Pydantic's BaseSettings.
    """
    # --- Core Application Settings ---
    LLM_PROVIDER: Literal["OpenAI", "Ollama", "Google"] = "Google"
    ENVIRONMENT: Literal["DEV", "PROD"] = "DEV"
    LOG_LEVEL: int = logging.INFO
    
    # --- Model-Specific Settings ---
    OLLAMA_MODEL: str = "tinyllama"
    GEMINI_MODEL: str = "gemini-1.5-flash" # Use the model you have access to

    # --- API Keys & Secrets ---
    OPENAI_API_KEY: Optional[SecretStr] = None
    GEMINI_API_KEY: Optional[SecretStr] = None

    # --- RAG System Configuration ---
    VECTOR_DB_PATH: str = "data/processed/faiss_index"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # --- Model Configuration for pydantic-settings ---
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra='ignore'
    )

    @field_validator("OPENAI_API_KEY", mode='after')
    @classmethod
    def validate_openai_key(cls, v: Optional[SecretStr], values) -> Optional[SecretStr]:
        if values.data.get("LLM_PROVIDER") == "OpenAI" and v is None:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER is 'OpenAI'")
        return v

    @field_validator("GEMINI_API_KEY", mode='after')
    @classmethod
    def validate_gemini_key(cls, v: Optional[SecretStr], values) -> Optional[SecretStr]:
        if values.data.get("LLM_PROVIDER") == "Google" and v is None:
            raise ValueError("GEMINI_API_KEY must be set when LLM_PROVIDER is 'Google'")
        return v

settings = Settings()
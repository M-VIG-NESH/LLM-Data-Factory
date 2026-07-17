"""
Configuration management using Pydantic Settings.
Loads environment variables and provides typed access to configuration.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    
    # Database
    database_url: str = Field(default="sqlite:///./data/llm_factory.db", env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Application
    app_name: str = Field(default="LLM Data Factory", env="APP_NAME")
    debug: bool = Field(default=True, env="DEBUG")
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    
    # LLM Settings
    default_llm_provider: Literal["groq", "gemini"] = Field(default="groq", env="DEFAULT_LLM_PROVIDER")
    default_model: str = Field(default="llama3-70b-8192", env="DEFAULT_MODEL")
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    # Chunking
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Generation
    max_qa_pairs_per_chunk: int = Field(default=3, env="MAX_QA_PAIRS_PER_CHUNK")
    min_answer_length: int = Field(default=5, env="MIN_ANSWER_LENGTH")
    
    # Paths
    upload_dir: str = Field(default="./data/uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="./data/processed", env="PROCESSED_DIR")
    export_dir: str = Field(default="./data/exports", env="EXPORT_DIR")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

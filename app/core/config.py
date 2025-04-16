from typing import Dict, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class Settings(BaseSettings):
    # LLM Provider 설정
    LLM_PROVIDER: LLMProvider = Field(default=LLMProvider.OLLAMA, env="LLM_PROVIDER")
    
    # Ollama 설정
    OLLAMA_URL: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    OLLAMA_MODEL: str = Field(default="gemma3:12b", env="OLLAMA_MODEL")
    OLLAMA_TIMEOUT: float = Field(default=30.0, env="OLLAMA_TIMEOUT")
    
    # OpenAI 설정
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4", env="OPENAI_MODEL")
    OPENAI_TIMEOUT: float = Field(default=30.0, env="OPENAI_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
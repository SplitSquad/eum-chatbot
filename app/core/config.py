from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # LLM 프로바이더 설정
    LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="사용할 LLM 프로바이더 (ollama 또는 openai)"
    )
    
    # Ollama 설정
    OLLAMA_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama API 서버 URL"
    )
    OLLAMA_MODEL: str = Field(
        default="gemma:2b",
        description="사용할 Ollama 모델"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=30,
        description="Ollama API 요청 타임아웃 (초)"
    )
    
    # OpenAI 설정
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API 키"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="사용할 OpenAI 모델"
    )
    OPENAI_TIMEOUT: int = Field(
        default=30,
        description="OpenAI API 요청 타임아웃 (초)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
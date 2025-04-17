from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # LLM 프로바이더 설정
    LIGHTWEIGHT_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="경량 모델용 LLM 프로바이더 (ollama 또는 openai)"
    )
    HIGH_PERFORMANCE_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA,
        description="고성능 모델용 LLM 프로바이더 (ollama 또는 openai)"
    )
    
    # Ollama 설정 (경량 모델)
    LIGHTWEIGHT_OLLAMA_URL: str = Field(
        default="http://localhost:11434",
        description="경량 모델용 Ollama API 서버 URL"
    )
    LIGHTWEIGHT_OLLAMA_MODEL: str = Field(
        default="gemma3:1b",
        description="경량 모델용 Ollama 모델"
    )
    LIGHTWEIGHT_OLLAMA_TIMEOUT: int = Field(
        default=30,
        description="경량 모델용 Ollama API 요청 타임아웃 (초)"
    )
    
    # Ollama 설정 (고성능 모델)
    HIGH_PERFORMANCE_OLLAMA_URL: str = Field(
        default="http://localhost:11434",
        description="고성능 모델용 Ollama API 서버 URL"
    )
    HIGH_PERFORMANCE_OLLAMA_MODEL: str = Field(
        default="gemma3:12b",
        description="고성능 모델용 Ollama 모델"
    )
    HIGH_PERFORMANCE_OLLAMA_TIMEOUT: int = Field(
        default=60,
        description="고성능 모델용 Ollama API 요청 타임아웃 (초)"
    )
    
    # OpenAI 설정 (경량 모델)
    LIGHTWEIGHT_OPENAI_API_KEY: str = Field(
        default="",
        description="경량 모델용 OpenAI API 키"
    )
    LIGHTWEIGHT_OPENAI_MODEL: str = Field(
        default="gpt-3.5-turbo",
        description="경량 모델용 OpenAI 모델"
    )
    LIGHTWEIGHT_OPENAI_TIMEOUT: int = Field(
        default=30,
        description="경량 모델용 OpenAI API 요청 타임아웃 (초)"
    )
    
    # OpenAI 설정 (고성능 모델)
    HIGH_PERFORMANCE_OPENAI_API_KEY: str = Field(
        default="",
        description="고성능 모델용 OpenAI API 키"
    )
    HIGH_PERFORMANCE_OPENAI_MODEL: str = Field(
        default="gpt-4",
        description="고성능 모델용 OpenAI 모델"
    )
    HIGH_PERFORMANCE_OPENAI_TIMEOUT: int = Field(
        default=60,
        description="고성능 모델용 OpenAI API 요청 타임아웃 (초)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 추가 필드를 무시하도록 설정

settings = Settings()
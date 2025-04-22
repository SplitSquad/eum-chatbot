from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    """

    # 기본 설정
    APP_NAME: str = Field(default="Eum Chatbot", description="애플리케이션 이름")
    APP_VERSION: str = Field(default="0.1.0", description="애플리케이션 버전")
    DEBUG: bool = Field(default=True, description="디버그 모드 활성화")
    API_PREFIX: str = Field(default="/api/v1", description="API 경로 prefix")
    SECRET_KEY: str = Field(default="SECRET_KEY", description="보안 키")

    # 경량 LLM 프로바이더 관련 설정
    LIGHTWEIGHT_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA, description="경량 LLM 제공자"
    )
    # 고성능 LLM 프로바이더 관련 설정
    HIGHPERFORMANCE_LLM_PROVIDER: LLMProvider = Field(
        default=LLMProvider.OLLAMA, description="고성능 LLM 제공자"
    )

    # Ollama 관련 설정
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434", description="Ollama API 기본 URL"
    )
    OLLAMA_LIGHTWEIGHT_MODEL: str = Field(
        default="llama3:8b", description="Ollama 경량 모델명"
    )
    OLLAMA_HIGHPERFORMANCE_MODEL: str = Field(
        default="gemma3:12b", description="Ollama 고성능 모델명"
    )

    # OpenAI 관련 설정
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API 키")
    OPENAI_LIGHTWEIGHT_MODEL: str = Field(
        default="gpt-3.5-turbo", description="OpenAI 경량 모델명"
    )
    OPENAI_HIGHPERFORMANCE_MODEL: str = Field(
        default="gpt-4", description="OpenAI 고성능 모델명"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings() 
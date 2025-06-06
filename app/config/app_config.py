from enum import Enum
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from loguru import logger
from typing import Any, Optional

# .env 파일 로드
load_dotenv()

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"


class Settings(BaseModel):
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
    LIGHTWEIGHT_LLM_PROVIDER: LLMProvider
    # 고성능 LLM 프로바이더 관련 설정
    HIGH_PERFORMANCE_LLM_PROVIDER: LLMProvider

    # Ollama 경량 모델 설정
    LIGHTWEIGHT_OLLAMA_URL: str
    LIGHTWEIGHT_OLLAMA_MODEL: str
    LIGHTWEIGHT_OLLAMA_TIMEOUT: int

    # Ollama 고성능 모델 설정
    HIGH_PERFORMANCE_OLLAMA_URL: str
    HIGH_PERFORMANCE_OLLAMA_MODEL: str
    HIGH_PERFORMANCE_OLLAMA_TIMEOUT: int

    # OpenAI 경량 모델 설정
    LIGHTWEIGHT_OPENAI_API_KEY: str
    LIGHTWEIGHT_OPENAI_MODEL: str
    LIGHTWEIGHT_OPENAI_TIMEOUT: int

    # OpenAI 고성능 모델 설정
    HIGH_PERFORMANCE_OPENAI_API_KEY: str
    HIGH_PERFORMANCE_OPENAI_MODEL: str
    HIGH_PERFORMANCE_OPENAI_TIMEOUT: int
    
    # Groq 설정
    GROQ_API_KEY: str
    GROQ_LIGHTWEIGHT_MODEL: str
    GROQ_HIGHPERFORMANCE_MODEL: str


def get_env_var(key: str, default_value: Optional[Any] = None) -> Any:
    """
    환경 변수를 가져옵니다.
    
    Args:
        key: 환경 변수 키
        default_value: 환경 변수가 없을 경우 반환할 기본값
        
    Returns:
        Any: 환경 변수 값 또는 기본값
    """
    try:
        value = os.getenv(key)
        if value is None:
            logger.warning(f"환경 변수 {key}가 설정되지 않았습니다. 기본값을 사용합니다: {default_value}")
            return default_value
        return value
    except Exception as e:
        logger.error(f"환경 변수 {key}를 가져오는 중 오류가 발생했습니다: {str(e)}")
        return default_value


# 설정 객체 생성
settings = Settings(
    # 경량 LLM 프로바이더 관련 설정
    LIGHTWEIGHT_LLM_PROVIDER=LLMProvider(get_env_var("LIGHTWEIGHT_LLM_PROVIDER", "groq")),
    # 고성능 LLM 프로바이더 관련 설정
    HIGH_PERFORMANCE_LLM_PROVIDER=LLMProvider(get_env_var("HIGH_PERFORMANCE_LLM_PROVIDER", "groq")),

    # Ollama 경량 모델 설정
    LIGHTWEIGHT_OLLAMA_URL=get_env_var("LIGHTWEIGHT_OLLAMA_URL", "http://localhost:11434"),
    LIGHTWEIGHT_OLLAMA_MODEL=get_env_var("LIGHTWEIGHT_OLLAMA_MODEL", "gemma3:1b"),
    LIGHTWEIGHT_OLLAMA_TIMEOUT=int(get_env_var("LIGHTWEIGHT_OLLAMA_TIMEOUT", "20")),

    # Ollama 고성능 모델 설정
    HIGH_PERFORMANCE_OLLAMA_URL=get_env_var("HIGH_PERFORMANCE_OLLAMA_URL", "http://localhost:11434"),
    HIGH_PERFORMANCE_OLLAMA_MODEL=get_env_var("HIGH_PERFORMANCE_OLLAMA_MODEL", "gemma3:4b"),
    HIGH_PERFORMANCE_OLLAMA_TIMEOUT=int(get_env_var("HIGH_PERFORMANCE_OLLAMA_TIMEOUT", "60")),

    # OpenAI 경량 모델 설정
    LIGHTWEIGHT_OPENAI_API_KEY=get_env_var("LIGHTWEIGHT_OPENAI_API_KEY", ""),
    LIGHTWEIGHT_OPENAI_MODEL=get_env_var("LIGHTWEIGHT_OPENAI_MODEL", "gpt-3.5-turbo"),
    LIGHTWEIGHT_OPENAI_TIMEOUT=int(get_env_var("LIGHTWEIGHT_OPENAI_TIMEOUT", "30")),

    # OpenAI 고성능 모델 설정
    HIGH_PERFORMANCE_OPENAI_API_KEY=get_env_var("HIGH_PERFORMANCE_OPENAI_API_KEY", ""),
    HIGH_PERFORMANCE_OPENAI_MODEL=get_env_var("HIGH_PERFORMANCE_OPENAI_MODEL", "gpt-4"),
    HIGH_PERFORMANCE_OPENAI_TIMEOUT=int(get_env_var("HIGH_PERFORMANCE_OPENAI_TIMEOUT", "60")),
    
    # Groq 설정
    GROQ_API_KEY=get_env_var("GROQ_API_KEY", ""),
    GROQ_LIGHTWEIGHT_MODEL=get_env_var("GROQ_LIGHTWEIGHT_MODEL", "llama-3.1-8b-instant"),
    GROQ_HIGHPERFORMANCE_MODEL=get_env_var("GROQ_HIGHPERFORMANCE_MODEL", "llama-3.3-70b-versatile"),
)

# 디버깅을 위한 설정 로그 출력
print(f"Loaded settings from .env: LIGHTWEIGHT_LLM_PROVIDER={settings.LIGHTWEIGHT_LLM_PROVIDER}") 
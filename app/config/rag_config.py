from typing import ClassVar, Dict
from enum import Enum
from pydantic import BaseModel

class RAGDomain(str, Enum):
    """RAG 도메인"""
    VISA_LAW = "visa_law"  # 비자/법률
    SOCIAL_SECURITY = "social_security"  # 사회보장제도
    TAX_FINANCE = "tax_finance"  # 세금/금융
    MEDICAL_HEALTH = "medical_health"  # 의료/건강
    EMPLOYMENT = "employment"  # 취업
    DAILY_LIFE = "daily_life"  # 일상생활

class RAGConfig(BaseModel):
    """RAG 설정"""
    
    # 벡터 DB 설정
    CHROMA_DB_PATH: str = "data/chroma"
    
    # 임베딩 모델 설정
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 검색 설정
    SEARCH_K: int = 3  # 검색할 문서 수
    SEARCH_THRESHOLD: float = 0.7  # 유사도 임계값
    
    # 도메인별 컬렉션 설정
    COLLECTIONS: ClassVar[Dict[RAGDomain, str]] = {
        RAGDomain.VISA_LAW: "vectorstore1",  # 비자/법률
        RAGDomain.SOCIAL_SECURITY: "vectorstore2",  # 사회보장제도
        RAGDomain.TAX_FINANCE: "vectorstore3",  # 세금/금융
        RAGDomain.MEDICAL_HEALTH: "vectorstore4",  # 의료/건강
        RAGDomain.EMPLOYMENT: "vectorstore5",  # 취업
        RAGDomain.DAILY_LIFE: "vectorstore6"  # 일상생활
    }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8" 
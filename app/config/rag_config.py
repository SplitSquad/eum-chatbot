from typing import ClassVar, Dict, Any
from enum import Enum
from pydantic import BaseModel
import os
from pathlib import Path

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
    # 기본 설정
    BASE_DIR: ClassVar[str] = os.getenv("BASE_DIR", str(Path(__file__).parent.parent.parent))
    CHROMA_DB_PATH: ClassVar[str] = os.path.join(BASE_DIR, "data", "chroma")
    EMBEDDING_MODEL: ClassVar[str] = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 768차원 모델로 변경
    SEARCH_K: ClassVar[int] = 5  # 검색 결과 수
    SEARCH_THRESHOLD: ClassVar[float] = 0.3  # 임계값 낮춤
    
    # 도메인별 설정
    DOMAIN_CONFIGS: ClassVar[Dict[RAGDomain, Dict[str, Any]]] = {
        RAGDomain.VISA_LAW: {
            "collection_name": "visa_law",
            "vectorstore_path": os.path.join(BASE_DIR, "data", "vectorstore1"),
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        RAGDomain.SOCIAL_SECURITY: {
            "collection_name": "social_security",
            "vectorstore_path": os.path.join(BASE_DIR, "data", "vectorstore2"),
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        RAGDomain.TAX_FINANCE: {
            "collection_name": "tax_finance",
            "vectorstore_path": os.path.join(BASE_DIR, "data", "vectorstore3"),
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        RAGDomain.MEDICAL_HEALTH: {
            "collection_name": "medical_health",
            "vectorstore_path": os.path.join(BASE_DIR, "data", "vectorstore4"),
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        RAGDomain.EMPLOYMENT: {
            "collection_name": "employment",
            "vectorstore_path": os.path.join(BASE_DIR, "data", "vectorstore5"),
            "chunk_size": 500,
            "chunk_overlap": 100
        },
        RAGDomain.DAILY_LIFE: {
            "collection_name": "daily_life",
            "vectorstore_path": os.path.join(BASE_DIR, "data", "vectorstore6"),
            "chunk_size": 500,
            "chunk_overlap": 100
        }
    }
    
    @classmethod
    def validate_paths(cls) -> None:
        """벡터 스토어 경로를 검증합니다."""
        for domain, config in cls.DOMAIN_CONFIGS.items():
            path = config["vectorstore_path"]
            if not os.path.exists(path):
                raise ValueError(f"벡터 스토어 경로가 존재하지 않습니다: {path}")
            if not os.path.exists(os.path.join(path, "chroma.sqlite3")):
                raise ValueError(f"ChromaDB 파일이 존재하지 않습니다: {path}/chroma.sqlite3")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8" 
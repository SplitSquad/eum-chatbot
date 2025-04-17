from typing import Dict, Any
from enum import Enum
from loguru import logger
from app.core.llm_client import get_llm_client

class QueryType(str, Enum):
    """질의 유형"""
    GENERAL = "general"  # 일반 대화
    REASONING = "reasoning"  # 추론
    WEB_SEARCH = "web_search"  # 웹 검색

class RAGType(str, Enum):
    """RAG 유형"""
    VISA_LAW = "visa_law"  # 비자/법률
    SOCIAL_SECURITY = "social_security"  # 사회보장제도
    TAX_FINANCE = "tax_finance"  # 세금/금융
    MEDICAL_HEALTH = "medical_health"  # 의료/건강
    EMPLOYMENT = "employment"  # 취업
    DAILY_LIFE = "daily_life"  # 일상생활
    NONE = "none"  # 적절한 RAG 없음

class Classifier:
    """분류 서비스"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
        logger.info(f"[분류] 경량 모델 사용: {self.llm_client.model}")
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """
        질의를 분류합니다.
        
        Args:
            query: 영어로 번역된 질의
            
        Returns:
            Dict[str, Any]: 분류 결과
        """
        try:
            logger.info(f"[분류] 질의 분류 시작: {query}")
            
            # 질의 유형 분류
            query_type = await self._classify_query_type(query)
            logger.info(f"[분류] 질의 유형: {query_type.value}")
            
            # RAG 유형 분류
            rag_type = await self._classify_rag_type(query)
            logger.info(f"[분류] RAG 유형: {rag_type.value}")
            
            return {
                "query": query,
                "query_type": query_type,
                "rag_type": rag_type
            }
        except Exception as e:
            logger.error(f"분류 중 오류 발생: {str(e)}")
            return {
                "query": query,
                "query_type": QueryType.GENERAL,
                "rag_type": RAGType.NONE
            }
    
    async def _classify_query_type(self, query: str) -> QueryType:
        """질의 유형을 분류합니다."""
        prompt = f"""
        Classify the following query into one of these types:
        - general: General conversation or simple information
        - reasoning: Requires logical reasoning or complex analysis
        - web_search: Requires up-to-date information from the web
        
        Query: {query}
        
        Return only the type name.
        """
        
        response = await self.llm_client.generate(prompt)
        response = response.strip().lower()
        
        try:
            return QueryType(response)
        except ValueError:
            return QueryType.GENERAL
    
    async def _classify_rag_type(self, query: str) -> RAGType:
        """RAG 유형을 분류합니다."""
        prompt = f"""
        Classify the following query into one of these RAG types:
        - visa_law: Visa and legal matters
        - social_security: Social security system
        - tax_finance: Tax and finance
        - medical_health: Medical and health
        - employment: Employment information
        - daily_life: Daily life
        - none: No appropriate RAG type
        
        Query: {query}
        
        Return only the type name.
        """
        
        response = await self.llm_client.generate(prompt)
        response = response.strip().lower()
        
        try:
            return RAGType(response)
        except ValueError:
            return RAGType.NONE 
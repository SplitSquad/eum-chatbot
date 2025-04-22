from typing import Dict, Any, Tuple
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

class ChatbotClassifier:
    """챗봇 분류기"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
        logger.info(f"[분류] 경량 모델 사용: {self.llm_client.model}")
    
    async def classify(self, query: str) -> Tuple[QueryType, RAGType]:
        """
        질의를 분류합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            Tuple[QueryType, RAGType]: 질의 유형과 RAG 유형
        """
        try:
            logger.info(f"[분류] 질의 분류 시작: {query}")
            
            # 질의 유형 분류
            query_type = await self._classify_query_type(query)
            logger.info(f"[분류] 질의 유형: {query_type.value}")
            
            # RAG 유형 분류
            rag_type = await self._classify_rag_type(query)
            logger.info(f"[분류] RAG 유형: {rag_type.value}")
            
            return query_type, rag_type
            
        except Exception as e:
            logger.error(f"분류 중 오류 발생: {str(e)}")
            return QueryType.GENERAL, RAGType.NONE
    
    async def _classify_query_type(self, query: str) -> QueryType:
        """질의 유형을 분류합니다."""
        try:
            # TODO: LLM을 활용한 질의 유형 분류 구현
            # 임시로 모든 질의를 일반 대화로 분류
            return QueryType.GENERAL
        except Exception as e:
            logger.error(f"질의 유형 분류 중 오류 발생: {str(e)}")
            return QueryType.GENERAL
    
    async def _classify_rag_type(self, query: str) -> RAGType:
        """RAG 유형을 분류합니다."""
        prompt = f"""
        Classify the following query into one of these RAG types:
        
        - visa_law: Questions about visas and legal matters, immigration, etc.
        - social_security: Questions about social security system, social insurance, etc.
        - tax_finance: Questions about taxes and finance, banking, etc.
        - medical_health: Questions about medical and health, medicine, etc.
        - employment: Questions about employment, job, etc.
        - daily_life: Questions about daily life, education, etc.
        - none: No specific domain knowledge required
        
        Query: {query}
        
        Return only the type name.
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            response = response.strip().lower()
            
            try:
                return RAGType(response)
            except ValueError:
                return RAGType.NONE
        except Exception as e:
            logger.error(f"RAG 유형 분류 중 오류 발생: {str(e)}")
            return RAGType.NONE 
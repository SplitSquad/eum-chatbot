from typing import Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.models.response_types import RAGType, ResponseType, QueryType

class QueryClassifier:
    """질의 분류기"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """
        질의를 분류합니다.
        
        Args:
            query: 입력 질의
            
        Returns:
            Dict[str, Any]: 분류 결과
        """
        # RAG 유형 분류
        rag_type = await self._classify_rag_type(query)
        
        # 응답 방식 분류
        response_type = await self._classify_response_type(query)
        
        # 질의 유형 분류
        query_type = await self._classify_query_type(query)
        
        return {
            "rag_type": rag_type,
            "response_type": response_type,
            "query_type": query_type
        }
    
    async def _classify_rag_type(self, query: str) -> RAGType:
        """RAG 유형을 분류합니다."""
        prompt = f"""
        다음 질문이 특정 도메인(예: 의학, 법률, 금융 등)에 대한 전문적인 지식이 필요한지 판단해주세요.
        질문: {query}
        
        응답 형식:
        - 도메인 특화 지식이 필요하면: domain_specific
        - 일반적인 지식으로 충분하면: none
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            return RAGType(response.strip().lower())
        except Exception as e:
            logger.error(f"RAG 유형 분류 중 오류 발생: {str(e)}")
            return RAGType.NONE
    
    async def _classify_response_type(self, query: str) -> ResponseType:
        """응답 방식을 분류합니다."""
        prompt = f"""
        다음 질문에 대해 가장 적절한 응답 방식을 선택해주세요.
        질문: {query}
        
        응답 형식:
        - 일반적인 대화나 간단한 정보 제공이면: general
        - 복잡한 추론이나 분석이 필요하면: reasoning
        - 최신 정보나 웹 검색이 필요하면: web_search
        - 특정 도메인의 전문 지식이 필요하면: rag
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            return ResponseType(response.strip().lower())
        except Exception as e:
            logger.error(f"응답 방식 분류 중 오류 발생: {str(e)}")
            return ResponseType.GENERAL
    
    async def _classify_query_type(self, query: str) -> QueryType:
        """질의 유형을 분류합니다."""
        prompt = f"""
        다음 질문의 유형을 분류해주세요.
        질문: {query}
        
        응답 형식:
        - 일반적인 질문이면: general
        - 추론이나 분석이 필요한 질문이면: reasoning
        - 검색이 필요한 질문이면: search
        - 특정 도메인 관련 질문이면: domain_specific
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            return QueryType(response.strip().lower())
        except Exception as e:
            logger.error(f"질의 유형 분류 중 오류 발생: {str(e)}")
            return QueryType.GENERAL 
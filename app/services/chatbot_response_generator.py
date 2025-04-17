from typing import Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.models.chatbot_response import ResponseType, RAGType
from app.services.chatbot_classifier import QueryClassifier

class ResponseGenerator:
    """응답 생성기"""
    
    def __init__(self):
        self.classifier = QueryClassifier()
        self.llm_client = get_llm_client(is_lightweight=True)
        self.high_performance_llm = get_llm_client(is_lightweight=False)
    
    async def generate_response(self, query: str) -> Dict[str, Any]:
        """
        질의에 대한 응답을 생성합니다.
        
        Args:
            query: 입력 질의
            
        Returns:
            Dict[str, Any]: 응답 결과
        """
        # 1. 질의 분류
        classification = await self.classifier.classify(query)
        
        # 2. 분류 결과에 따른 응답 생성
        if classification["response_type"] == ResponseType.GENERAL:
            return await self._generate_general_response(query)
        elif classification["response_type"] == ResponseType.REASONING:
            return await self._generate_reasoning_response(query)
        elif classification["response_type"] == ResponseType.WEB_SEARCH:
            return await self._generate_web_search_response(query)
        elif classification["response_type"] == ResponseType.RAG:
            return await self._generate_rag_response(query, classification["rag_type"])
        else:
            return await self._generate_general_response(query)
    
    async def _generate_general_response(self, query: str) -> Dict[str, Any]:
        """일반 응답을 생성합니다."""
        try:
            response = await self.llm_client.generate(query)
            return {
                "type": "general",
                "response": response,
                "metadata": {
                    "model": "lightweight",
                    "response_type": "general"
                }
            }
        except Exception as e:
            logger.error(f"일반 응답 생성 중 오류 발생: {str(e)}")
            return {
                "type": "error",
                "response": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {"error": str(e)}
            }
    
    async def _generate_reasoning_response(self, query: str) -> Dict[str, Any]:
        """추론 응답을 생성합니다."""
        try:
            # Chain of Thought 프롬프트
            cot_prompt = f"""
            다음 질문에 대해 단계별로 추론하여 답변해주세요.
            
            질문: {query}
            
            답변 형식:
            1. 문제 이해
            2. 필요한 정보 분석
            3. 추론 과정
            4. 최종 답변
            """
            
            response = await self.high_performance_llm.generate(cot_prompt)
            return {
                "type": "reasoning",
                "response": response,
                "metadata": {
                    "model": "high_performance",
                    "response_type": "reasoning",
                    "method": "chain_of_thought"
                }
            }
        except Exception as e:
            logger.error(f"추론 응답 생성 중 오류 발생: {str(e)}")
            return {
                "type": "error",
                "response": "죄송합니다. 추론 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {"error": str(e)}
            }
    
    async def _generate_web_search_response(self, query: str) -> Dict[str, Any]:
        """웹 검색 응답을 생성합니다."""
        try:
            # TODO: WebSearch Tool 구현 필요
            # 임시로 일반 응답으로 대체
            return await self._generate_general_response(query)
        except Exception as e:
            logger.error(f"웹 검색 응답 생성 중 오류 발생: {str(e)}")
            return {
                "type": "error",
                "response": "죄송합니다. 웹 검색 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {"error": str(e)}
            }
    
    async def _generate_rag_response(self, query: str, rag_type: RAGType) -> Dict[str, Any]:
        """RAG 응답을 생성합니다."""
        try:
            # TODO: RAG 구현 필요
            # 임시로 일반 응답으로 대체
            return await self._generate_general_response(query)
        except Exception as e:
            logger.error(f"RAG 응답 생성 중 오류 발생: {str(e)}")
            return {
                "type": "error",
                "response": "죄송합니다. RAG 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {"error": str(e)}
            } 
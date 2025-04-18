from typing import Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.services.chatbot.chatbot_classifier import QueryType, RAGType
from app.services.chatbot.chatbot_rag_service import ChatbotRAGService

class ChatbotResponseGenerator:
    """챗봇 응답 생성기"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
        self.high_performance_llm = get_llm_client(is_lightweight=False)
        self.rag_service = ChatbotRAGService()
    
    async def generate_response(self, query: str, query_type: QueryType, rag_type: RAGType) -> str:
        """응답을 생성합니다."""
        try:
            if query_type == QueryType.REASONING:
                return await self._generate_reasoning_response(query, rag_type)
            elif query_type == QueryType.WEB_SEARCH:
                return await self._generate_web_search_response(query, rag_type)
            elif query_type == QueryType.GENERAL:
                return await self._generate_general_response(query, rag_type)
            else:
                return "죄송합니다. 현재는 일반 대화, 추론, 웹 검색 유형의 질문만 처리할 수 있습니다."
                
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
    
    async def _generate_simple_response(self, query: str) -> Dict[str, Any]:
        """단순 응답을 생성합니다."""
        try:
            response = await self.llm_client.generate(query)
            return {
                "type": "simple",
                "response": response,
                "metadata": {
                    "model": "lightweight",
                    "response_type": "simple"
                }
            }
        except Exception as e:
            logger.error(f"단순 응답 생성 중 오류 발생: {str(e)}")
            return {
                "type": "error",
                "response": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {"error": str(e)}
            }
    
    async def _generate_reasoning_response(self, query: str, rag_type: RAGType) -> str:
        """
        추론 응답을 생성합니다.
        
        Args:
            query: 질의
            rag_type: RAG 유형
            
        Returns:
            str: 생성된 응답
        """
        try:
            logger.info("[Response] 추론 응답 생성 시작")
            
            # RAG 컨텍스트 가져오기
            context = await self.rag_service.get_context(rag_type, query)
            if not context:
                logger.warning("[Response] RAG 컨텍스트가 없습니다.")
                return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."
            
            # 프롬프트 생성
            prompt = f"""다음은 외국인을 위한 한국 생활 정보입니다:

{context}

질문: {query}

위 정보를 바탕으로 질문에 답변해주세요. 답변은 친절하고 이해하기 쉽게 작성해주세요."""
            
            # 응답 생성
            response = await self.high_performance_llm.generate(prompt)
            if not response:
                logger.error("[Response] LLM 응답 생성 실패")
                return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
            
            logger.info("[Response] 추론 응답 생성 완료")
            return response
            
        except Exception as e:
            logger.error(f"추론 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
    
    async def _generate_web_search_response(self, query: str) -> Dict[str, Any]:
        """웹 검색 응답을 생성합니다."""
        try:
            # TODO: WebSearch Tool 구현 필요
            # 임시로 단순 응답으로 대체
            return await self._generate_simple_response(query)
        except Exception as e:
            logger.error(f"웹 검색 응답 생성 중 오류 발생: {str(e)}")
            return {
                "type": "error",
                "response": "죄송합니다. 웹 검색 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {"error": str(e)}
            } 
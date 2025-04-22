from typing import Optional, Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.services.common.preprocessor import translate_query
from app.services.chatbot.chatbot_classifier import ChatbotClassifier, QueryType, RAGType
from app.services.chatbot.chatbot_rag_service import ChatbotRAGService
from app.services.common.postprocessor import Postprocessor

class Chatbot:
    """챗봇 클래스"""
    
    def __init__(self):
        self.classifier = ChatbotClassifier()
        self.rag_service = ChatbotRAGService()
        self.postprocessor = Postprocessor()
        self.llm_client = get_llm_client(is_lightweight=True)
        self.high_performance_llm = get_llm_client(is_lightweight=False)
        logger.info(f"[챗봇] 고성능 모델 사용: {self.high_performance_llm.model}")
    
    async def get_response(self, query: str, uid: str) -> Dict[str, Any]:
        """질의에 대한 응답을 생성합니다."""
        try:
            # 질의 유형과 RAG 유형 분류
            query_type, rag_type = await self.classifier.classify(query)
            
            # 응답 생성
            response = await self.generate_response(query, query_type, rag_type)
            
            return {
                "response": response,
                "metadata": {
                    "query": query,
                    "query_type": query_type.value,
                    "rag_type": rag_type.value,
                    "uid": uid
                }
            }
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return {
                "response": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {
                    "query": query,
                    "query_type": "error",
                    "rag_type": "error",
                    "uid": uid,
                    "error": str(e)
                }
            }
    
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
    
    async def _generate_general_response(self, query: str, rag_type: RAGType) -> str:
        """일반 대화 응답을 생성합니다."""
        try:
            # RAG 컨텍스트 생성
            context = ""
            if rag_type != RAGType.NONE:
                context = await self.rag_service.get_context(rag_type, query)
                if context:
                    logger.info(f"[챗봇] RAG 컨텍스트 생성 완료: {len(context)}자")
            
            # 프롬프트 생성
            prompt = self._generate_prompt(query, context)
            logger.info("[챗봇] 프롬프트 생성 완료")
            
            # 응답 생성
            logger.info("[챗봇] 응답 생성 시작")
            response = await self.llm_client.generate(prompt)
            logger.info(f"[챗봇] 응답 생성 완료: {len(response)}자")
            
            return response.strip()
        except Exception as e:
            logger.error(f"일반 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
    
    async def _generate_reasoning_response(self, query: str, rag_type: RAGType) -> str:
        """추론 응답을 생성합니다."""
        try:
            logger.info("[챗봇] 추론 응답 생성 시작")
            
            # RAG 컨텍스트 가져오기
            context = await self.rag_service.get_context(rag_type, query)
            if not context:
                logger.warning("[챗봇] RAG 컨텍스트가 없습니다.")
                return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."
            
            # 프롬프트 생성
            prompt = f"""Here is information about living in Korea for foreigners:

{context}

Question: {query}

Please answer the question based on the information above. Make your response friendly and easy to understand."""
            
            # 응답 생성
            response = await self.high_performance_llm.generate(prompt)
            if not response:
                logger.error("[챗봇] LLM 응답 생성 실패")
                return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
            
            logger.info("[챗봇] 추론 응답 생성 완료")
            return response.strip()
            
        except Exception as e:
            logger.error(f"추론 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
    
    async def _generate_web_search_response(self, query: str, rag_type: RAGType) -> str:
        """웹 검색 응답을 생성합니다."""
        # TODO: 웹 검색 응답 로직 구현
        return "죄송합니다. 웹 검색 응답 기능은 아직 구현 중입니다."
    
    def _generate_prompt(self, query: str, context: str = "") -> str:
        """프롬프트를 생성합니다."""
        base_prompt = f"""
        Please provide a helpful response to the following query:
        Query: {query}
        """
        
        if context:
            base_prompt += f"""
            
            Here is some relevant context:
            {context}
            """
        
        base_prompt += """
        
        Please provide a clear and concise response in Korean.
        """
        
        return base_prompt 
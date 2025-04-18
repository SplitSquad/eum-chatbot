from typing import Optional, Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.services.chatbot.chatbot_classifier import ChatbotClassifier, QueryType, RAGType
from app.services.chatbot.chatbot_rag_service import ChatbotRAGService
from app.services.common.translator import Translator
from app.services.chatbot.chatbot_response_generator import ChatbotResponseGenerator
from app.models.chatbot_response import ChatbotResponse, ChatbotResponseData

class ChatbotService:
    """챗봇 서비스"""
    
    def __init__(self):
        self.classifier = ChatbotClassifier()
        self.translator = Translator()
        self.response_generator = ChatbotResponseGenerator()
        
    async def process_query(self, query: str, uid: str) -> ChatbotResponse:
        """
        사용자 질의를 처리합니다.
        
        Args:
            query: 사용자 질의
            uid: 사용자 ID
            
        Returns:
            ChatbotResponse: 챗봇 응답
        """
        try:
            logger.info(f"[Chatbot] 질의 처리 시작 - 사용자: {uid}")
            
            # 1. 질의 분류
            query_type, rag_type = await self.classifier.classify(query)
            logger.info(f"[Chatbot] 질의 분류 결과 - 유형: {query_type.value}, RAG 유형: {rag_type.value}")
            
            # 2. 영어 번역
            english_query = await self.translator.translate(query)
            logger.info(f"[Chatbot] 영어 번역 결과: {english_query}")
            
            # 3. 응답 생성
            response = await self.response_generator.generate_response(query, query_type, rag_type)
            logger.info("[Chatbot] 응답 생성 완료")
            
            # 4. 응답 반환
            return ChatbotResponse(
                response=response,
                data=ChatbotResponseData(
                    original_query=query,
                    english_query=english_query,
                    source_lang="tr",
                    query_type=query_type,
                    rag_type=rag_type,
                    used_rag=rag_type != RAGType.NONE,
                    metadata={
                        "source_lang": "tr",
                        "query_type": query_type.value,
                        "rag_type": rag_type.value,
                        "model": "gemma3:12b"
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"질의 처리 중 오류 발생: {str(e)}")
            return ChatbotResponse(
                response="죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                data=ChatbotResponseData(
                    original_query=query,
                    english_query="",
                    source_lang="tr",
                    query_type=QueryType.SIMPLE,
                    rag_type=RAGType.NONE,
                    used_rag=False,
                    metadata={
                        "error": str(e),
                        "model": "gemma3:12b"
                    }
                )
            ) 
from typing import Optional, Dict, Any
from loguru import logger
from app.services.chatbot.chatbot_classifier import ChatbotClassifier, QueryType, RAGType
from app.services.chatbot.chatbot_response_generator import ChatbotResponseGenerator

class Chatbot:
    """챗봇 클래스 - 워크플로우 관리"""
    
    def __init__(self):
        self.classifier = ChatbotClassifier()
        self.response_generator = ChatbotResponseGenerator()
        logger.info("[챗봇] 초기화 완료")
    
    async def get_response(self, query: str, uid: str) -> Dict[str, Any]:
        """질의에 대한 응답을 생성합니다."""
        try:
            # 질의 유형과 RAG 유형 분류
            query_type, rag_type = await self.classifier.classify(query)
            logger.info(f"[챗봇] 분류 완료 - 질의 유형: {query_type.value}, RAG 유형: {rag_type.value}")
            
            # 응답 생성기를 통해 응답 생성
            response = await self.response_generator.generate_response(query, query_type, rag_type)
            logger.info("[챗봇] 응답 생성 완료")
            
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
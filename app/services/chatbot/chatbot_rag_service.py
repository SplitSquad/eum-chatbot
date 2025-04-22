from typing import List
from loguru import logger
from app.services.common.rag_service import RAGService
from app.services.chatbot.chatbot_classifier import RAGType

class ChatbotRAGService(RAGService):
    """챗봇 전용 RAG 서비스"""
    
    async def get_context(self, query: str, rag_type: RAGType) -> str:
        """
        질의와 RAG 유형에 맞는 컨텍스트를 가져옵니다.
        
        Args:
            query: 사용자 질의
            rag_type: RAG 유형
            
        Returns:
            str: 생성된 컨텍스트
        """
        try:
            # 부모 클래스의 get_context 메서드 호출
            context = await super().get_context(rag_type, query)
            
            # 챗봇용 컨텍스트 후처리
            if context:
                logger.info(f"[챗봇 RAG] 컨텍스트 생성 완료: {len(context)}자")
                # 필요한 경우 여기에 챗봇 전용 컨텍스트 처리 로직 추가
            else:
                logger.info("[챗봇 RAG] 컨텍스트를 찾을 수 없습니다.")
            
            return context
        except Exception as e:
            logger.error(f"챗봇 RAG 컨텍스트 생성 중 오류 발생: {str(e)}")
            return "" 
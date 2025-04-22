from typing import Dict, Any, Optional
import os
from loguru import logger
import torch
from app.core.llm_client import get_llm_client
from app.services.chatbot.chatbot_classifier import QueryType, RAGType
from app.services.common.rag_service import RAGService

class ChatbotResponseGenerator:
    """챗봇 응답 생성기"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.llm_client = get_llm_client(is_lightweight=True)
        self.high_performance_llm = get_llm_client(is_lightweight=False)
        logger.info(f"[응답 생성기] 고성능 모델 사용: {self.high_performance_llm.model}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[응답 생성기] 디바이스: {self.device}")
    
    async def generate_response(self, query: str, query_type: QueryType, rag_type: RAGType) -> str:
        """
        질의에 대한 응답을 생성합니다.
        
        Args:
            query: 사용자 질의
            query_type: 질의 유형
            rag_type: RAG 유형
            
        Returns:
            str: 생성된 응답
        """
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
                    logger.info(f"[응답 생성기] RAG 컨텍스트 생성 완료: {len(context)}자")
            
            # 프롬프트 생성
            prompt = self._generate_prompt(query, context)
            logger.info("[응답 생성기] 프롬프트 생성 완료")
            
            # 응답 생성
            logger.info("[응답 생성기] 응답 생성 시작")
            response = await self.llm_client.generate(prompt)
            logger.info(f"[응답 생성기] 응답 생성 완료: {len(response)}자")
            
            return response.strip()
        except Exception as e:
            logger.error(f"일반 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
    
    async def _generate_reasoning_response(self, query: str, rag_type: RAGType) -> str:
        """추론 응답을 생성합니다."""
        try:
            logger.info("[응답 생성기] 추론 응답 생성 시작")
            
            # RAG 컨텍스트 가져오기
            context = await self.rag_service.get_context(rag_type, query)
            if not context:
                logger.warning("[응답 생성기] RAG 컨텍스트가 없습니다.")
                return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."
            
            # 프롬프트 생성
            prompt = f"""다음은 외국인을 위한 한국 생활 정보입니다:

{context}

질문: {query}

위 정보를 바탕으로 질문에 답변해주세요. 답변은 친절하고 이해하기 쉽게 작성해주세요."""
            
            # 응답 생성
            response = await self.high_performance_llm.generate(prompt)
            if not response:
                logger.error("[응답 생성기] LLM 응답 생성 실패")
                return "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다."
            
            logger.info("[응답 생성기] 추론 응답 생성 완료")
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
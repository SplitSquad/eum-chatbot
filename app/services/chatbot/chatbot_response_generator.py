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
        # 고성능 모델로 Groq API 사용
        self.high_performance_llm = get_llm_client(is_lightweight=False)
        logger.info(f"[응답 생성기] Groq 고성능 모델 사용: {self.high_performance_llm.model}, 타임아웃: {self.high_performance_llm.timeout}초")
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
            logger.info(f"[RESPONSE] Input query (English): {query}")
            logger.info(f"[RESPONSE] Query type: {query_type.value}")
            logger.info(f"[RESPONSE] RAG type: {rag_type.value}")
            
            if query_type == QueryType.REASONING:
                return await self._generate_reasoning_response(query, rag_type)
            elif query_type == QueryType.WEB_SEARCH:
                return await self._generate_web_search_response(query, rag_type)
            elif query_type == QueryType.GENERAL:
                return await self._generate_general_response(query, rag_type)
            else:
                return "Sorry, currently only general conversation, reasoning, and web search type questions can be processed."
                
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return "Sorry, an error occurred while generating the response."
    
    async def _generate_general_response(self, query: str, rag_type: RAGType) -> str:
        """일반 대화 응답을 생성합니다."""
        try:
            # RAG 컨텍스트 생성
            context = ""
            if rag_type != RAGType.NONE:
                context = await self.rag_service.get_context(rag_type, query)
                if context:
                    logger.info(f"[응답 생성기] RAG 컨텍스트 생성 완료: {len(context)}자")
                    logger.debug(f"[RESPONSE] RAG context: {context[:200]}...")
                else:
                    logger.info("[RESPONSE] No RAG context available")
            
            # 프롬프트 생성
            prompt = self._generate_prompt(query, context)
            logger.info("[응답 생성기] 프롬프트 생성 완료")
            logger.debug(f"[RESPONSE] Generated prompt: {prompt}")
            
            # 응답 생성 (Groq 고성능 모델 사용)
            logger.info(f"[응답 생성기] 응답 생성 시작 (타임아웃: {self.high_performance_llm.timeout}초)")
            response = await self.high_performance_llm.generate(prompt)
            logger.info(f"[응답 생성기] 응답 생성 완료: {len(response)}자")
            logger.info(f"[RESPONSE] Generated response: {response}")
            
            return response.strip()
        except Exception as e:
            logger.error(f"일반 응답 생성 중 오류 발생: {str(e)}")
            # 타임아웃 오류일 경우 실제 타임아웃 값을 포함한 메시지 반환
            if "타임아웃" in str(e):
                return f"Sorry, the response generation timed out after {self.high_performance_llm.timeout} seconds. Please try again later."
            return "Sorry, an error occurred while generating the response."
    
    async def _generate_reasoning_response(self, query: str, rag_type: RAGType) -> str:
        """추론 응답을 생성합니다."""
        try:
            logger.info("[응답 생성기] 추론 응답 생성 시작")
            
            # RAG 컨텍스트 가져오기
            context = await self.rag_service.get_context(rag_type, query)
            if not context:
                logger.warning("[응답 생성기] RAG 컨텍스트가 없습니다.")
                logger.info("[RESPONSE] No RAG context available for reasoning")
                return "Sorry, no information could be found for this question."
            else:
                logger.debug(f"[RESPONSE] RAG context for reasoning: {context[:200]}...")
            
            # 프롬프트 생성
            prompt = f"""The following is information about life in Korea for foreigners:

{context}

Question: {query}

Please answer the question based on the information above. Provide a friendly and easy-to-understand response."""
            
            logger.debug(f"[RESPONSE] Reasoning prompt: {prompt[:200]}...")
            
            # 응답 생성 (Groq 고성능 모델 사용)
            logger.info(f"[응답 생성기] 추론 응답 생성 시작 (타임아웃: {self.high_performance_llm.timeout}초)")
            response = await self.high_performance_llm.generate(prompt)
            if not response:
                logger.error("[응답 생성기] LLM 응답 생성 실패")
                return "Sorry, an error occurred while generating the response."
            
            logger.info("[응답 생성기] 추론 응답 생성 완료")
            logger.info(f"[RESPONSE] Generated reasoning response: {response}")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"추론 응답 생성 중 오류 발생: {str(e)}")
            # 타임아웃 오류일 경우 실제 타임아웃 값을 포함한 메시지 반환
            if "타임아웃" in str(e):
                return f"Sorry, the response generation timed out after {self.high_performance_llm.timeout} seconds. Please try again later."
            return "Sorry, an error occurred while generating the response."
    
    async def _generate_web_search_response(self, query: str, rag_type: RAGType) -> str:
        """웹 검색 응답을 생성합니다."""
        try:
            logger.info("[응답 생성기] 웹 검색 응답 생성 시작")
            
            # 웹 검색 결과는 현재 구현되어 있지 않으므로 설명만 제공
            prompt = f"""Please respond to the following query that requires up-to-date information:

Question: {query}

Please note that real-time web search is not yet available and explain this limitation politely."""
            
            # 응답 생성 (Groq 고성능 모델 사용)
            logger.info(f"[응답 생성기] 웹 검색 응답 생성 시작 (타임아웃: {self.high_performance_llm.timeout}초)")
            response = await self.high_performance_llm.generate(prompt)
            
            logger.info("[응답 생성기] 웹 검색 응답 생성 완료")
            logger.info(f"[RESPONSE] Generated web search response: {response}")
            
            return response.strip()
        except Exception as e:
            logger.error(f"웹 검색 응답 생성 중 오류 발생: {str(e)}")
            return "Sorry, web search response functionality is still under development."
    
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
        
        Please provide a clear and concise response in English.
        """
        
        return base_prompt 
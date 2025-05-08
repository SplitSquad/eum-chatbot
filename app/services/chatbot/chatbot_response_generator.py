from typing import Dict, Any, Optional
import os
from loguru import logger
import torch
from app.core.llm_client import get_llm_client
from app.services.chatbot.chatbot_classifier import QueryType, RAGType
from app.services.common.rag_service import RAGService
from app.services.common.web_search_service import WebSearchService
from app.services.common.postprocessor import Postprocessor

class ChatbotResponseGenerator:
    """챗봇 응답 생성기"""
    
    def __init__(self):
        self.rag_service = RAGService()
        self.web_search_service = WebSearchService()
        self.postprocessor = Postprocessor()
        # 고성능 모델로 Groq API 사용
        self.high_performance_llm = get_llm_client(is_lightweight=False)
        logger.info(f"[응답 생성기] Groq 고성능 모델 사용: {self.high_performance_llm.model}, 타임아웃: {self.high_performance_llm.timeout}초")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[응답 생성기] 디바이스: {self.device}")
    
    async def generate_response(self, query: str, query_type: QueryType, rag_type: RAGType, lang_code: str) -> str:
        """
        질의에 대한 응답을 생성합니다.
        
        Args:
            query: 사용자 질의
            query_type: 질의 유형
            rag_type: RAG 유형
            lang_code: 언어 코드
            
        Returns:
            str: 생성된 응답
        """
        try:
            logger.info(f"[RESPONSE] Input query (English): {query}")
            logger.info(f"[RESPONSE] Query type: {query_type.value}")
            logger.info(f"[RESPONSE] RAG type: {rag_type.value}")
            logger.info(f"[RESPONSE] Language code: {lang_code}")
            
            if query_type == QueryType.REASONING:
                return await self._generate_reasoning_response(query, rag_type, lang_code)
            elif query_type == QueryType.WEB_SEARCH:
                return await self._generate_web_search_response(query, rag_type, lang_code)
            elif query_type == QueryType.GENERAL:
                return await self._generate_general_response(query, rag_type, lang_code)
            else:
                return "Sorry, currently only general conversation, reasoning, and web search type questions can be processed."
                
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return "Sorry, an error occurred while generating the response."
    
    async def _generate_general_response(self, query: str, rag_type: RAGType, lang_code: str) -> str:
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
            
            # 후처리 적용
            if response:
                postprocessed = await self.postprocessor.postprocess(
                    response=response,
                    source_lang=lang_code,
                    rag_type=rag_type.value
                )
                response = postprocessed["response"]
            
            logger.info(f"[응답 생성기] 응답 생성 완료: {len(response)}자")
            logger.info(f"[RESPONSE] Generated response: {response}")
            
            return response.strip()
        except Exception as e:
            logger.error(f"일반 응답 생성 중 오류 발생: {str(e)}")
            # 타임아웃 오류일 경우 실제 타임아웃 값을 포함한 메시지 반환
            if "타임아웃" in str(e):
                return f"Sorry, the response generation timed out after {self.high_performance_llm.timeout} seconds. Please try again later."
            return "Sorry, an error occurred while generating the response."
    
    async def _generate_reasoning_response(self, query: str, rag_type: RAGType, lang_code: str) -> str:
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
            
            # 후처리 적용
            if response:
                postprocessed = await self.postprocessor.postprocess(
                    response=response,
                    source_lang=lang_code,
                    rag_type=rag_type.value
                )
                response = postprocessed["response"]
            
            logger.info("[응답 생성기] 추론 응답 생성 완료")
            logger.info(f"[RESPONSE] Generated reasoning response: {response}")
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"추론 응답 생성 중 오류 발생: {str(e)}")
            # 타임아웃 오류일 경우 실제 타임아웃 값을 포함한 메시지 반환
            if "타임아웃" in str(e):
                return f"Sorry, the response generation timed out after {self.high_performance_llm.timeout} seconds. Please try again later."
            return "Sorry, an error occurred while generating the response."
    
    async def _generate_web_search_response(self, query: str, rag_type: RAGType, lang_code: str) -> str:
        """웹 검색 응답을 생성합니다."""
        try:
            logger.info("[응답 생성기] 웹 검색 응답 생성 시작")
            logger.info(f"[응답 생성기] 검색 질의: {query}")
            logger.info(f"[응답 생성기] 언어 코드: {lang_code}")
            
            # 웹 검색 실행
            web_context = await self.web_search_service.get_context(query)
            logger.info(f"[응답 생성기] 웹 검색 컨텍스트 생성 완료: {len(web_context) if web_context else 0}자")
            
            # RAG 컨텍스트도 함께 사용 (있는 경우)
            rag_context = ""
            if rag_type != RAGType.NONE:
                rag_context = await self.rag_service.get_context(rag_type, query)
                if rag_context:
                    logger.info(f"[응답 생성기] RAG 컨텍스트 생성 완료: {len(rag_context)}자")
            
            # 컨텍스트 결합
            context = ""
            if web_context and rag_context:
                context = f"웹 검색 결과:\n{web_context}\n\n추가 정보:\n{rag_context}"
            elif web_context:
                context = f"웹 검색 결과:\n{web_context}"
            elif rag_context:
                context = f"추가 정보:\n{rag_context}"
            
            if not context:
                logger.warning("[응답 생성기] 검색 결과가 없습니다.")
                return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."
            
            # 프롬프트 생성
            prompt = f"""다음 정보를 바탕으로 질문에 대한 포괄적인 답변을 제공해주세요:

웹 및 RAG 검색 결과: {context}

질문: {query}"""
            
            # 응답 생성 (Groq 고성능 모델 사용)
            logger.info(f"[응답 생성기] 웹 검색 응답 생성 시작 (타임아웃: {self.high_performance_llm.timeout}초)")
            response = await self.high_performance_llm.generate(prompt)
            
            # 후처리 적용 (언어 코드와 RAG 타입 전달)
            if response:
                postprocessed = await self.postprocessor.postprocess(
                    response=response,
                    source_lang=lang_code,  # preprocessor에서 추출한 언어 코드 사용
                    rag_type=rag_type.value
                )
                response = postprocessed["response"]
            
            logger.info("[응답 생성기] 웹 검색 응답 생성 완료")
            logger.info(f"[응답 생성기] 생성된 응답: {response[:200]}...")
            
            return response.strip()
        except Exception as e:
            logger.error(f"웹 검색 응답 생성 중 오류 발생: {str(e)}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
    
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
from typing import Optional, Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.services.preprocessor import translate_query
from app.services.classifier import Classifier, QueryType, RAGType
from app.services.rag_service import RAGService
from app.services.postprocessor import Postprocessor

class Chatbot:
    """챗봇 클래스"""
    
    def __init__(self):
        self.classifier = Classifier()
        self.rag_service = RAGService()
        self.postprocessor = Postprocessor()
        self.llm_client = get_llm_client(is_lightweight=False)
        logger.info(f"[챗봇] 고성능 모델 사용: {self.llm_client.model}")
    
    async def get_response(self, query: str) -> Dict[str, Any]:
        """
        사용자 질의에 대한 응답을 생성합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            Dict[str, Any]: 응답 결과
        """
        try:
            logger.info(f"[챗봇] 질의 처리 시작: {query}")
            
            # 1. 전처리
            translation = await translate_query(query)
            english_query = translation["translated_query"]
            source_lang = translation["lang_code"]
            logger.info(f"[챗봇] 전처리 완료 - 영어: {english_query}, 원문 언어: {source_lang}")
            
            # 2. 분류
            classification = await self.classifier.classify(english_query)
            query_type = classification["query_type"]
            rag_type = classification["rag_type"]
            logger.info(f"[챗봇] 분류 완료 - 질의 유형: {query_type.value}, RAG 유형: {rag_type.value}")
            
            # 3. 응답 생성
            if query_type == QueryType.GENERAL:
                response = await self._generate_general_response(english_query, rag_type)
            elif query_type == QueryType.REASONING:
                response = await self._generate_reasoning_response(english_query, rag_type)
            elif query_type == QueryType.WEB_SEARCH:
                response = await self._generate_web_search_response(english_query, rag_type)
            else:
                response = "죄송합니다. 현재는 일반 대화, 추론, 웹 검색 유형의 질문만 처리할 수 있습니다."
            
            # 4. 후처리
            postprocessed = await self.postprocessor.postprocess(response, source_lang, rag_type.value)
            logger.info(f"[챗봇] 후처리 완료 - RAG 사용: {postprocessed['used_rag']}")
            
            # 5. 결과 반환
            return {
                "response": postprocessed["response"],
                "data": {
                    "original_query": query,
                    "english_query": english_query,
                    "source_lang": source_lang,
                    "query_type": query_type.value,
                    "rag_type": postprocessed["rag_type"],
                    "used_rag": postprocessed["used_rag"],
                    "metadata": {
                        "model": self.llm_client.model,
                        "response_type": query_type.value
                    }
                }
            }
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return {
                "response": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "data": {
                    "original_query": query,
                    "response": None,
                    "metadata": {
                        "model": self.llm_client.model,
                        "response_type": "error"
                    }
                }
            }
    
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
        # TODO: 추론 응답 로직 구현
        return "죄송합니다. 추론 응답 기능은 아직 구현 중입니다."
    
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
from typing import Optional, Dict, Any
from loguru import logger
from app.services.chatbot.chatbot_classifier import ChatbotClassifier, QueryType, RAGType
from app.services.chatbot.chatbot_response_generator import ChatbotResponseGenerator
from app.services.common.postprocessor import Postprocessor
from app.services.common.preprocessor import translate_query

class Chatbot:
    """챗봇 클래스 - 워크플로우 관리"""
    
    def __init__(self):
        self.classifier = ChatbotClassifier()
        self.response_generator = ChatbotResponseGenerator()
        self.postprocessor = Postprocessor()
        logger.info("[챗봇] 초기화 완료")
    
    async def get_response(self, query: str, uid: str) -> Dict[str, Any]:
        """질의에 대한 응답을 생성합니다."""
        try:
            logger.info(f"[WORKFLOW] ====== Starting chatbot workflow for user {uid} ======")
            logger.info(f"[WORKFLOW] Original query: {query}")
            
            # 질의 유형과 RAG 유형 분류
            query_type, rag_type = await self.classifier.classify(query)
            logger.info(f"[챗봇] 분류 완료 - 질의 유형: {query_type.value}, RAG 유형: {rag_type.value}")
            logger.info(f"[WORKFLOW] Query classified as {query_type.value}, RAG type: {rag_type.value}")
            
            # 언어 감지 및 번역 (preprocessor 사용)
            logger.info(f"[WORKFLOW] Step 1: Preprocessing (language detection and translation)")
            translation_result = await translate_query(query)
            source_lang = translation_result["lang_code"]
            english_query = translation_result["translated_query"]
            logger.info(f"[챗봇] 언어 감지 완료 - 소스 언어: {source_lang}, 영어 번역: {english_query}")
            logger.info(f"[WORKFLOW] Preprocessing complete: source_lang={source_lang}, english_query='{english_query}'")
            
            # 응답 생성기를 통해 응답 생성
            logger.info(f"[WORKFLOW] Step 2: Response generation")
            response = await self.response_generator.generate_response(english_query, query_type, rag_type)
            logger.info("[챗봇] 응답 생성 완료")
            logger.info(f"[WORKFLOW] Response generation complete: '{response}'")
            
            # 후처리 (원문 언어로 번역)
            logger.info(f"[WORKFLOW] Step 3: Postprocessing (translation back to original language)")
            processed_response = await self.postprocessor.postprocess(response, source_lang, rag_type.value)
            logger.info("[챗봇] 후처리 완료")
            logger.info(f"[WORKFLOW] Postprocessing complete: '{processed_response['response']}'")
            
            result = {
                "response": processed_response["response"],
                "metadata": {
                    "query": query,
                    "english_query": english_query,
                    "query_type": query_type.value,
                    "rag_type": rag_type.value,
                    "source_lang": source_lang,
                    "uid": uid,
                    "used_rag": processed_response["used_rag"]
                }
            }
            
            logger.info(f"[WORKFLOW] ====== Chatbot workflow completed for user {uid} ======")
            return result
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            logger.error(f"[WORKFLOW] ====== Error in chatbot workflow: {str(e)} ======")
            return {
                "response": "Sorry, an error occurred while generating the response.",
                "metadata": {
                    "query": query,
                    "query_type": "error",
                    "rag_type": "error",
                    "uid": uid,
                    "error": str(e)
                }
            }
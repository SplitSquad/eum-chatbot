# app/api/v1/chatbot.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
from app.services.chatbot.chatbot_classifier import ChatbotClassifier, QueryType, RAGType
from app.services.chatbot.chatbot_response_generator import ChatbotResponseGenerator
from app.services.common.preprocessor import translate_query

router = APIRouter(
    prefix="/chatbot",
    tags=["Chatbot"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"}
    }
)

class ChatbotRequest(BaseModel):
    """챗봇 요청 모델"""
    query: str
    uid: str

class ChatbotResponse(BaseModel):
    """챗봇 응답 모델"""
    response: str
    metadata: Dict[str, Any]

@router.post(
    "",
    response_model=ChatbotResponse,
    summary="챗봇 응답 생성",
    description="사용자 질의에 대한 챗봇 응답을 생성합니다."
)
async def chatbot_handler(request: ChatbotRequest) -> ChatbotResponse:
    """
    챗봇 핸들러
    
    Args:
        request: 챗봇 요청
        
    Returns:
        ChatbotResponse: 챗봇 응답
        
    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        # 분류기와 응답 생성기 초기화
        classifier = ChatbotClassifier()
        response_generator = ChatbotResponseGenerator()
        
        # 언어 감지 및 번역
        logger.info(f"[API] 언어 감지 및 번역 시작: {request.query}")
        translation_result = await translate_query(request.query)
        source_lang = translation_result["lang_code"]
        english_query = translation_result["translated_query"]
        logger.info(f"[API] 언어 감지 및 번역 완료 - 소스 언어: {source_lang}, 영어 번역: {english_query}")
        
        # 질의 분류
        query_type, rag_type = await classifier.classify(english_query)
        logger.info(f"[API] 질의 분류 결과 - 유형: {query_type.value}, RAG: {rag_type.value}")
        
        # 응답 생성
        response = await response_generator.generate_response(english_query, query_type, rag_type, source_lang)
        logger.info(f"[API] 응답 생성 완료: {len(response)}자")
        
        # 응답 반환
        return ChatbotResponse(
            response=response,
            metadata={
                "query_type": query_type.value,
                "rag_type": rag_type.value,
                "uid": request.uid,
                "source_lang": source_lang,
                "english_query": english_query
            }
        )
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"챗봇 처리 중 오류가 발생했습니다: {str(e)}"
        )

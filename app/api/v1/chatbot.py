# app/api/v1/chatbot.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
from app.services.chatbot_response_generator import ResponseGenerator

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

class ChatbotRequest(BaseModel):
    """챗봇 요청 모델"""
    uid: str
    query: str

class ChatbotResponse(BaseModel):
    """챗봇 응답 모델"""
    response: str
    data: Dict[str, Any]

@router.post("", response_model=ChatbotResponse)
async def chatbot_handler(request: ChatbotRequest) -> ChatbotResponse:
    """
    챗봇 핸들러
    
    Args:
        request: 챗봇 요청
        
    Returns:
        ChatbotResponse: 챗봇 응답
    """
    try:
        # 응답 생성기 초기화
        generator = ResponseGenerator()
        
        # 응답 생성
        result = await generator.generate_response(request.query)
        
        # 응답 형식 변환
        return ChatbotResponse(
            response=result["type"],
            data={
                "original_query": request.query,
                "response": result["response"],
                "metadata": result["metadata"]
            }
        )
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"챗봇 처리 중 오류가 발생했습니다: {str(e)}"
        )

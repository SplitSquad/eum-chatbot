# app/api/v1/chatbot.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
from app.services.chatbot.chatbot import Chatbot

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
    data: Dict[str, Any]

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
        # 챗봇 초기화
        chatbot = Chatbot()
        
        # 응답 생성
        result = await chatbot.get_response(request.query)
        
        # 응답 반환
        return ChatbotResponse(
            response=result["response"],
            data=result["data"]
        )
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"챗봇 처리 중 오류가 발생했습니다: {str(e)}"
        )

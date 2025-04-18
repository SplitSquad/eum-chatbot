# app/api/v1/agentic.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
from app.services.agentic.agentic_classifier import AgenticClassifier
from app.services.agentic.agentic_response_generator import AgenticResponseGenerator

router = APIRouter(
    prefix="/agentic",
    tags=["Agentic"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"}
    }
)

class AgenticRequest(BaseModel):
    """에이전틱 요청 모델"""
    query: str
    uid: str

class AgenticResponse(BaseModel):
    """에이전틱 응답 모델"""
    response: str
    metadata: Dict[str, Any]

@router.post(
    "",
    response_model=AgenticResponse,
    summary="에이전틱 응답 생성",
    description="사용자 질의에 대한 에이전틱 응답을 생성합니다."
)
async def agentic_handler(request: AgenticRequest) -> AgenticResponse:
    """
    에이전틱 핸들러
    
    Args:
        request: 에이전틱 요청
        
    Returns:
        AgenticResponse: 에이전틱 응답
        
    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        # 분류기 초기화
        classifier = AgenticClassifier()
        
        # 응답 생성기 초기화
        generator = AgenticResponseGenerator()
        
        # 질의 분류
        agentic_type = await classifier.classify(request.query)
        
        # 응답 생성
        result = await generator.generate_response(request.query, agentic_type)
        
        # 응답 반환
        return AgenticResponse(
            response=result["response"],
            metadata=result["metadata"]
        )
    except Exception as e:
        logger.error(f"에이전틱 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"에이전틱 처리 중 오류가 발생했습니다: {str(e)}"
        )

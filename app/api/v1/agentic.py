# app/api/v1/agentic.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from loguru import logger
from app.services.common.preprocessor import translate_query
from app.services.agentic.agentic_classifier import AgentClassifier

router = APIRouter(
    prefix="/agentic",
    tags=["Agentic"],
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"}
    }
)

class AgenticRequest(BaseModel):
    """에이전트 요청 모델"""
    query: str
    uid: str

class AgenticResponse(BaseModel):
    """에이전트 응답 모델"""
    response: str
    data: Dict

@router.post(
    "",
    response_model=AgenticResponse,
    summary="에이전트 분류",
    description="사용자 질의에 대한 에이전트 유형, 액션 유형, 도메인을 분류합니다."
)
async def agentic_handler(payload: AgenticRequest) -> AgenticResponse:
    """
    에이전트 핸들러
    
    Args:
        payload: 에이전트 요청
        
    Returns:
        AgenticResponse: 에이전트 분류 결과
        
    Raises:
        HTTPException: 처리 중 오류가 발생한 경우
    """
    try:
        # 번역
        translated = await translate_query(payload.query)
        english_query = translated["translated_query"]
        lang_code = translated["lang_code"]
        
        # 분류
        classifier = AgentClassifier()
        agentic_result = await classifier.classify(english_query)
        
        return AgenticResponse(
            response="agentic_classification",
            data={
                "original_query": payload.query,
                "lang_code": lang_code,
                "translated_query": english_query,
                "agentic_classification": {
                    "agent_type": agentic_result["agent_type"].value,
                    "action_type": agentic_result["action_type"].value,
                    "domain": agentic_result["domain"].value
                }
            }
        )
    except Exception as e:
        logger.error(f"에이전트 처리 중 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"에이전트 처리 중 오류가 발생했습니다: {str(e)}"
        )

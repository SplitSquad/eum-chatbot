# app/api/v1/classify_agentic.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.core.classify_agentic import classify_agentic_query

router = APIRouter(prefix="/classify", tags=["Classification"])

class ClassificationRequest(BaseModel):
    uid: str
    query: str

@router.post("/agentic")
async def classify_agentic(payload: ClassificationRequest) -> Dict:
    result = await classify_agentic_query(payload.query)
    return {
        "response": "agentic_classified",
        "data": result
    }

# app/api/v1/classify_chatbot.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.core.classify_chatbot import classify_domain_query

router = APIRouter(prefix="/classify", tags=["Classification"])

class ClassificationRequest(BaseModel):
    uid: str
    query: str

@router.post("/chatbot")
async def classify_chatbot(payload: ClassificationRequest) -> Dict:
    result = await classify_domain_query(payload.query)
    return {
        "response": "chatbot_classified",
        "data": result
    }

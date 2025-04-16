# app/api/v1/agentic.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.core.preprocess import translate_query
from app.core.classify_chatbot import classify_agentic_query

router = APIRouter(prefix="/agentic", tags=["Agentic"])

class AgenticRequest(BaseModel):
    uid: str
    query: str

@router.post("")
async def agentic_handler(payload: AgenticRequest) -> Dict:
    translated = await translate_query(payload.query)
    english_query = translated["translated_query"]
    lang_code = translated["lang_code"]

    agentic_result = await classify_agentic_query(english_query)

    return {
        "response": "agentic_classification",
        "data": {
            "original_query": payload.query,
            "lang_code": lang_code,
            "translated_query": english_query,
            "agentic_classification": agentic_result
        }
    }

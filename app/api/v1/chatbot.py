# app/api/v1/chatbot.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict
from app.core.preprocess import translate_query
from app.core.classify_chatbot import classify_specialized_query

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

class ChatbotRequest(BaseModel):
    uid: str
    query: str

@router.post("")
async def chatbot_handler(payload: ChatbotRequest) -> Dict:
    translated = await translate_query(payload.query)
    english_query = translated["translated_query"]
    lang_code = translated["lang_code"]

    specialized_result = await classify_specialized_query(english_query)

    return {
        "response": "specialized_classification",
        "data": {
            "original_query": payload.query,
            "lang_code": lang_code,
            "translated_query": english_query,
            "specialized_classification": specialized_result
        }
    }

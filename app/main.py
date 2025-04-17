# app/main.py

from fastapi import FastAPI
from app.api.v1 import chatbot, agentic

app = FastAPI(
    title="EUM Chatbot API",
    description="EUM Chatbot API 서비스",
    version="1.0.0"
)

# API 라우터 등록
app.include_router(chatbot.router, prefix="/api/v1")
app.include_router(agentic.router, prefix="/api/v1")

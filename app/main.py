# app/main.py

from fastapi import FastAPI, Request
from app.api.v1 import chatbot
from app.config.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Setup logging
logger = setup_logging()
logger.info("Application starting up...")

app = FastAPI(
    title="EUM Chatbot API",
    description="EUM Chatbot API 서비스",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# API 라우터 등록
app.include_router(chatbot.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("[WORKFLOW] Server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[WORKFLOW] Server shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

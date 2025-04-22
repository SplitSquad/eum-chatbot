# app/main.py

from fastapi import FastAPI
from app.api.v1 import chatbot, agentic
from app.config.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware

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
app.include_router(agentic.router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("[WORKFLOW] Server started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("[WORKFLOW] Server shutting down")

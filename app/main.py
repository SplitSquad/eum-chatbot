# app/main.py

from fastapi import FastAPI
from app.api.v1 import agentic, chatbot, preprocess, classify_agentic, classify_chatbot

app = FastAPI()

app.include_router(agentic.router)
app.include_router(chatbot.router)
app.include_router(preprocess.router)
app.include_router(classify_agentic.router)
app.include_router(classify_chatbot.router)

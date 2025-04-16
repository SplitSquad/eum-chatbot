# app/main.py

from fastapi import FastAPI
from app.api.v1 import agentic, chatbot, preprocess, classify

app = FastAPI()

app.include_router(agentic.router)
app.include_router(chatbot.router)
app.include_router(preprocess.router)
app.include_router(classify.router)

## 📚 2. 전문 질의 분류기 (RAG)

# app/core/classify_chatbot.py

import httpx
from typing import Dict
import json
from app.core.config import OLLAMA_URL, MODEL_NAME

DOMAIN_PROMPT_TEMPLATE = """
You're a query classifier. Classify the query into whether it needs a specialized RAG (Retrieval-Augmented Generation),
identify which domain it belongs to, and whether it's a general, reasoning, or search-type query.

Return in this **exact** JSON format:

```json
{{
  "needs_rag": true/false,
  "rag_domain": "none" or "visa" or "finance" or "medical" or "employment" or "housing" or "culture",
  "query_type": "general" or "reasoning" or "search"
}}

Query: "{query}" """

async def classify_domain_query(query: str) -> Dict[str, str]: payload = { "model": MODEL_NAME, "prompt": DOMAIN_PROMPT_TEMPLATE.format(query=query), "stream": False, }

async with httpx.AsyncClient() as client:
    response = await client.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    result = response.json()["response"]

json_str = result.strip().strip("```json").strip("```").strip()

try:
    return json.loads(json_str)
except Exception as e:
    raise ValueError(f"[전문 분류기] 파싱 오류: {e}\n원본 응답: {result}")

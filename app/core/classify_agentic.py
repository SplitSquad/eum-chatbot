# app/core/classify_agentic.py

import httpx
from typing import Dict
import json
from app.core.config import OLLAMA_URL, MODEL_NAME

AGENTIC_PROMPT_TEMPLATE = """
You're an agent classifier. Given the user query, determine whether it requires an agentic action,
what kind of agent is needed (e.g., booking, consultation, callback), and whether a professional-level answer is required.

Return in this **exact** JSON format:

```json
{{
  "needs_agent": true/false,
  "agent_type": "none" or "booking" or "consultation" or "callback",
  "requires_expert": true/false
}}
Query: "{query}" """

async def classify_agentic_query(query: str) -> Dict[str, str]: payload = { "model": MODEL_NAME, "prompt": AGENTIC_PROMPT_TEMPLATE.format(query=query), "stream": False, }

async with httpx.AsyncClient() as client:
    response = await client.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    result = response.json()["response"]

json_str = result.strip().strip("```json").strip("```").strip()

try:
    return json.loads(json_str)
except Exception as e:
    raise ValueError(f"[Agentic 분류기] 파싱 오류: {e}\n원본 응답: {result}")

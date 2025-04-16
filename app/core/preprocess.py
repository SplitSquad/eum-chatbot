# app/core/preprocess.py

import httpx
from typing import Dict
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b"  # 또는 mistral, llama3 등

PROMPT_TEMPLATE = """
Detect the language of the following query, then translate it to English.

Return the result ONLY in this JSON format:

```json
{{
  "translated_query": "...",
  "lang_code": "..."
}}
Query: "{query}" """

# app/core/preprocess.py

import httpx
from typing import Dict
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b"  # 또는 mistral, llama3 등

PROMPT_TEMPLATE = """
Detect the language of the following query, then translate it to English.

Return the result ONLY in this JSON format:

```json
{{
  "translated_query": "...",
  "lang_code": "..."
}}
Query: "{query}" """


async def translate_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리를 영어로 번역하고 언어 코드를 반환합니다.

    Args:
        query (str):번역할 쿼리문장

    Returns:
        Dict[str, str]: 키 'translated_query'와 'lang_code'가 포함된 딕셔너리
                        - 'translated_query': 영어로 번역된 문장
                        - 'lang_code': 원래 언어의 IETF 코드 (예: 'ko' 한국어)

    Raises:
        ValueError:번역 응답을 파싱할 수 없는 경우에 예외를 발생시킵니다.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": PROMPT_TEMPLATE.format(query=query),
        "stream": False,
    }

    with httpx.Client() as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()["response"]

    # JSON 추출 및 처리
    json_str = result.strip().strip("```json").strip("```").strip()

    try:
        data = json.loads(json_str)
        return {
            "translated_query": data["translated_query"],
            "lang_code": data["lang_code"]
        }
    except Exception as e:
        raise ValueError(f"번역 응답 파싱 오류: {e}\n원본 응답: {result}")
## 📚 2. 전문 질의 분류기 (RAG)

# app/core/classify_chatbot.py

from typing import Dict
import json
from loguru import logger
from app.core.llm_client import get_llm_client

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

AGENTIC_PROMPT_TEMPLATE = """
You're a query classifier. Determine if the query requires agentic capabilities (the ability to take actions or make decisions).

Return in this **exact** JSON format:

```json
{{
  "is_agentic": true/false,
  "agentic_type": "none" or "action" or "decision" or "planning"
}}

Query: "{query}" """

SPECIALIZED_PROMPT_TEMPLATE = """
You're a query classifier. Determine if the query requires specialized knowledge or expertise.

Return in this **exact** JSON format:

```json
{{
  "is_specialized": true/false,
  "specialization_type": "none" or "technical" or "professional" or "academic"
}}

Query: "{query}" """

async def classify_domain_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리를 분류하여 RAG가 필요한지, 어떤 도메인에 속하는지, 그리고 쿼리 타입을 결정합니다.

    Args:
        query (str): 분류할 쿼리문장

    Returns:
        Dict[str, str]: 분류 결과를 포함한 딕셔너리
    """
    try:
        # LLM 클라이언트 초기화
        llm_client = get_llm_client()
        
        # 도메인 분류 요청
        result = await llm_client.generate(
            prompt=DOMAIN_PROMPT_TEMPLATE.format(query=query)
        )
        
        # JSON 추출 및 처리
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[도메인 분류기] 응답: {json_str}")
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[도메인 분류기] JSON 파싱 실패: {str(e)}")
        logger.error(f"[도메인 분류기] 원본 응답: {result}")
        raise ValueError(f"[도메인 분류기] 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        logger.error(f"[도메인 분류기] 처리 중 오류 발생: {str(e)}")
        raise ValueError(f"[도메인 분류기] 처리 중 오류 발생: {str(e)}")

async def classify_agentic_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리가 에이전트 능력이 필요한지 분류합니다.

    Args:
        query (str): 분류할 쿼리문장

    Returns:
        Dict[str, str]: 분류 결과를 포함한 딕셔너리
    """
    try:
        # LLM 클라이언트 초기화
        llm_client = get_llm_client()
        
        # 에이전트 분류 요청
        result = await llm_client.generate(
            prompt=AGENTIC_PROMPT_TEMPLATE.format(query=query)
        )
        
        # JSON 추출 및 처리
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[에이전트 분류기] 응답: {json_str}")
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[에이전트 분류기] JSON 파싱 실패: {str(e)}")
        logger.error(f"[에이전트 분류기] 원본 응답: {result}")
        raise ValueError(f"[에이전트 분류기] 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        logger.error(f"[에이전트 분류기] 처리 중 오류 발생: {str(e)}")
        raise ValueError(f"[에이전트 분류기] 처리 중 오류 발생: {str(e)}")

async def classify_specialized_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리가 전문 지식이 필요한지 분류합니다.

    Args:
        query (str): 분류할 쿼리문장

    Returns:
        Dict[str, str]: 분류 결과를 포함한 딕셔너리
    """
    try:
        # LLM 클라이언트 초기화
        llm_client = get_llm_client()
        
        # 전문 분류 요청
        result = await llm_client.generate(
            prompt=SPECIALIZED_PROMPT_TEMPLATE.format(query=query)
        )
        
        # JSON 추출 및 처리
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[전문 분류기] 응답: {json_str}")
        
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"[전문 분류기] JSON 파싱 실패: {str(e)}")
        logger.error(f"[전문 분류기] 원본 응답: {result}")
        raise ValueError(f"[전문 분류기] 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        logger.error(f"[전문 분류기] 처리 중 오류 발생: {str(e)}")
        raise ValueError(f"[전문 분류기] 처리 중 오류 발생: {str(e)}")

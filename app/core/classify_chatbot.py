## 📚 2. 전문 질의 분류기 (RAG)

# app/core/classify_chatbot.py

from typing import Dict
import json
import time
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
    start_time = time.time()
    
    try:
        # 경량 모델 클라이언트 초기화
        init_start = time.time()
        llm_client = get_llm_client(is_lightweight=True)
        init_time = time.time() - init_start
        logger.info(f"[Domain] 클라이언트 초기화 시간: {init_time:.2f}초")
        
        # 서버 연결 확인
        conn_start = time.time()
        if not await llm_client.check_connection():
            raise ConnectionError("LLM 서버 연결 실패")
        conn_time = time.time() - conn_start
        logger.info(f"[Domain] 서버 연결 확인 시간: {conn_time:.2f}초")
        
        # 도메인 분류 요청
        gen_start = time.time()
        result = await llm_client.generate(
            prompt=DOMAIN_PROMPT_TEMPLATE.format(query=query)
        )
        gen_time = time.time() - gen_start
        logger.info(f"[Domain] LLM 생성 시간: {gen_time:.2f}초")
        
        # JSON 추출 및 처리
        parse_start = time.time()
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[Domain] 응답: {json_str}")
        
        data = json.loads(json_str)
        parse_time = time.time() - parse_start
        logger.info(f"[Domain] JSON 파싱 시간: {parse_time:.2f}초")
        
        total_time = time.time() - start_time
        logger.info(f"[Domain] 전체 실행 시간: {total_time:.2f}초")
        
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[Domain] JSON 파싱 실패: {str(e)}")
        logger.error(f"[Domain] 원본 응답: {result}")
        raise ValueError(f"[Domain] 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Domain] 처리 중 오류 발생: {str(e)} (소요 시간: {elapsed:.2f}초)")
        raise ValueError(f"[Domain] 처리 중 오류 발생: {str(e)}")

async def classify_agentic_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리가 에이전트 능력이 필요한지 분류합니다.

    Args:
        query (str): 분류할 쿼리문장

    Returns:
        Dict[str, str]: 분류 결과를 포함한 딕셔너리
    """
    start_time = time.time()
    
    try:
        # 경량 모델 클라이언트 초기화
        init_start = time.time()
        llm_client = get_llm_client(is_lightweight=True)
        init_time = time.time() - init_start
        logger.info(f"[Agentic] 클라이언트 초기화 시간: {init_time:.2f}초")
        
        # 서버 연결 확인
        conn_start = time.time()
        if not await llm_client.check_connection():
            raise ConnectionError("LLM 서버 연결 실패")
        conn_time = time.time() - conn_start
        logger.info(f"[Agentic] 서버 연결 확인 시간: {conn_time:.2f}초")
        
        # 에이전트 분류 요청
        gen_start = time.time()
        result = await llm_client.generate(
            prompt=AGENTIC_PROMPT_TEMPLATE.format(query=query)
        )
        gen_time = time.time() - gen_start
        logger.info(f"[Agentic] LLM 생성 시간: {gen_time:.2f}초")
        
        # JSON 추출 및 처리
        parse_start = time.time()
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[Agentic] 응답: {json_str}")
        
        data = json.loads(json_str)
        parse_time = time.time() - parse_start
        logger.info(f"[Agentic] JSON 파싱 시간: {parse_time:.2f}초")
        
        total_time = time.time() - start_time
        logger.info(f"[Agentic] 전체 실행 시간: {total_time:.2f}초")
        
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[Agentic] JSON 파싱 실패: {str(e)}")
        logger.error(f"[Agentic] 원본 응답: {result}")
        raise ValueError(f"[Agentic] 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Agentic] 처리 중 오류 발생: {str(e)} (소요 시간: {elapsed:.2f}초)")
        raise ValueError(f"[Agentic] 처리 중 오류 발생: {str(e)}")

async def classify_specialized_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리가 전문 지식이 필요한지 분류합니다.

    Args:
        query (str): 분류할 쿼리문장

    Returns:
        Dict[str, str]: 분류 결과를 포함한 딕셔너리
    """
    start_time = time.time()
    
    try:
        # 경량 모델 클라이언트 초기화
        init_start = time.time()
        llm_client = get_llm_client(is_lightweight=True)
        init_time = time.time() - init_start
        logger.info(f"[Specialized] 클라이언트 초기화 시간: {init_time:.2f}초")
        
        # 서버 연결 확인
        conn_start = time.time()
        if not await llm_client.check_connection():
            raise ConnectionError("LLM 서버 연결 실패")
        conn_time = time.time() - conn_start
        logger.info(f"[Specialized] 서버 연결 확인 시간: {conn_time:.2f}초")
        
        # 전문 분류 요청
        gen_start = time.time()
        result = await llm_client.generate(
            prompt=SPECIALIZED_PROMPT_TEMPLATE.format(query=query)
        )
        gen_time = time.time() - gen_start
        logger.info(f"[Specialized] LLM 생성 시간: {gen_time:.2f}초")
        
        # JSON 추출 및 처리
        parse_start = time.time()
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[Specialized] 응답: {json_str}")
        
        data = json.loads(json_str)
        parse_time = time.time() - parse_start
        logger.info(f"[Specialized] JSON 파싱 시간: {parse_time:.2f}초")
        
        total_time = time.time() - start_time
        logger.info(f"[Specialized] 전체 실행 시간: {total_time:.2f}초")
        
        return data
    except json.JSONDecodeError as e:
        logger.error(f"[Specialized] JSON 파싱 실패: {str(e)}")
        logger.error(f"[Specialized] 원본 응답: {result}")
        raise ValueError(f"[Specialized] 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Specialized] 처리 중 오류 발생: {str(e)} (소요 시간: {elapsed:.2f}초)")
        raise ValueError(f"[Specialized] 처리 중 오류 발생: {str(e)}")

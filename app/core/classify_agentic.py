# app/core/classify_agentic.py

from typing import Dict
import json
import time
from loguru import logger
from app.core.llm_client import get_llm_client

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

async def classify_agentic_query(query: str) -> Dict[str, str]:
    """
    주어진 쿼리가 에이전트 능력이 필요한지 분류합니다.

    Args:
        query (str): 분류할 쿼리문장

    Returns:
        Dict[str, str]: 분류 결과를 포함한 딕셔너리

    Raises:
        ConnectionError: LLM 서버 연결 실패 시
        TimeoutError: 요청 타임아웃 시
        ValueError: 응답 파싱 실패 시
    """
    start_time = time.time()
    logger.info(f"[Agentic 분류기] 입력 쿼리: {query}")
    
    # LLM 클라이언트 초기화
    llm_client = get_llm_client()
    init_time = time.time()
    logger.info(f"[Agentic 분류기] 클라이언트 초기화 시간: {init_time - start_time:.2f}초")
    
    # 서버 연결 확인
    connection_start = time.time()
    if not await llm_client.check_connection():
        raise ConnectionError("LLM 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    connection_time = time.time() - connection_start
    logger.info(f"[Agentic 분류기] 서버 연결 확인 시간: {connection_time:.2f}초")
    
    try:
        # LLM 요청 및 응답 처리
        generate_start = time.time()
        result = await llm_client.generate(
            prompt=AGENTIC_PROMPT_TEMPLATE.format(query=query)
        )
        generate_time = time.time() - generate_start
        logger.info(f"[Agentic 분류기] LLM 생성 시간: {generate_time:.2f}초")
        logger.debug(f"[Agentic 분류기] 원본 응답: {result}")

        # JSON 파싱
        parse_start = time.time()
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[Agentic 분류기] 파싱된 JSON 문자열: {json_str}")

        parsed_result = json.loads(json_str)
        parse_time = time.time() - parse_start
        logger.info(f"[Agentic 분류기] JSON 파싱 시간: {parse_time:.2f}초")
        logger.info(f"[Agentic 분류기] 최종 분류 결과: {json.dumps(parsed_result, ensure_ascii=False)}")
        
        total_time = time.time() - start_time
        logger.info(f"[Agentic 분류기] 총 실행 시간: {total_time:.2f}초")
        
        return parsed_result
    except json.JSONDecodeError as e:
        logger.error(f"[Agentic 분류기] JSON 파싱 실패: {str(e)}")
        logger.error(f"[Agentic 분류기] 원본 응답: {result}")
        raise ValueError(f"응답 JSON 파싱 실패: {str(e)}")
    except Exception as e:
        logger.error(f"[Agentic 분류기] 예상치 못한 오류: {str(e)}")
        logger.error(f"[Agentic 분류기] 원본 응답: {result}")
        raise ValueError(f"처리 중 오류 발생: {str(e)}")

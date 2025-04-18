# app/services/common/preprocessor.py

from typing import Dict
import json
import time
from loguru import logger
from app.core.llm_client import get_llm_client

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
    start_time = time.time()
    result = None
    
    try:
        # 경량 모델 클라이언트 초기화
        init_start = time.time()
        llm_client = get_llm_client(is_lightweight=True)
        init_time = time.time() - init_start
        logger.info(f"[Preprocess] 클라이언트 초기화 시간: {init_time:.2f}초")
        logger.info(f"[Preprocess] 사용 모델: {llm_client.model}")
        
        # 서버 연결 확인
        conn_start = time.time()
        if not await llm_client.check_connection():
            raise ConnectionError("LLM 서버 연결 실패")
        conn_time = time.time() - conn_start
        logger.info(f"[Preprocess] 서버 연결 확인 시간: {conn_time:.2f}초")
        
        # 번역 요청
        gen_start = time.time()
        result = await llm_client.generate(
            prompt=PROMPT_TEMPLATE.format(query=query)
        )
        gen_time = time.time() - gen_start
        logger.info(f"[Preprocess] LLM 생성 시간: {gen_time:.2f}초")
        
        # JSON 추출 및 처리
        parse_start = time.time()
        json_str = result.strip().strip("```json").strip("```").strip()
        logger.debug(f"[Preprocess] 번역 응답: {json_str}")
        
        data = json.loads(json_str)
        parse_time = time.time() - parse_start
        logger.info(f"[Preprocess] JSON 파싱 시간: {parse_time:.2f}초")
        
        total_time = time.time() - start_time
        logger.info(f"[Preprocess] 전체 실행 시간: {total_time:.2f}초")
        
        return {
            "translated_query": data["translated_query"],
            "lang_code": data["lang_code"]
        }
    except json.JSONDecodeError as e:
        logger.error(f"[Preprocess] 번역 응답 JSON 파싱 실패: {str(e)}")
        logger.error(f"[Preprocess] 원본 응답: {result}")
        raise ValueError(f"번역 응답 파싱 오류: {e}\n원본 응답: {result}")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Preprocess] 번역 처리 중 오류 발생: {str(e)} (소요 시간: {elapsed:.2f}초)")
        raise ValueError(f"번역 처리 중 오류 발생: {str(e)}") 
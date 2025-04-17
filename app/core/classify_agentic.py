# app/core/classify_agentic.py

import json
import time
from loguru import logger
from app.core.llm_client import get_llm_client

async def classify_agentic_query(query: str) -> dict:
    """
    사용자의 쿼리를 에이전트성 쿼리로 분류합니다.
    
    Args:
        query (str): 사용자의 쿼리
        
    Returns:
        dict: 분류 결과
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
        
        # 프롬프트 생성
        prompt = f"""
        다음 쿼리를 분석하여 에이전트성 쿼리인지 분류해주세요.
        에이전트성 쿼리는 사용자가 특정 작업을 수행하도록 요청하는 쿼리입니다.
        
        쿼리: {query}
        
        JSON 형식으로 응답해주세요:
        {{
            "is_agentic": boolean,  // 에이전트성 쿼리 여부
            "confidence": float,    // 분류 신뢰도 (0.0 ~ 1.0)
            "reason": string        // 분류 이유
        }}
        """
        
        # LLM 응답 생성
        gen_start = time.time()
        response = await llm_client.generate(prompt)
        gen_time = time.time() - gen_start
        logger.info(f"[Agentic] LLM 생성 시간: {gen_time:.2f}초")
        
        # JSON 파싱
        parse_start = time.time()
        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"[Agentic] JSON 파싱 실패: {str(e)}")
            logger.error(f"[Agentic] 원본 응답: {response}")
            raise ValueError(f"LLM 응답 파싱 실패: {str(e)}")
        parse_time = time.time() - parse_start
        logger.info(f"[Agentic] JSON 파싱 시간: {parse_time:.2f}초")
        
        # 결과 검증
        if not isinstance(result, dict):
            raise ValueError("LLM 응답이 올바른 JSON 형식이 아닙니다.")
        
        required_fields = ["is_agentic", "confidence", "reason"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"LLM 응답에 필수 필드 '{field}'가 없습니다.")
        
        total_time = time.time() - start_time
        logger.info(f"[Agentic] 전체 실행 시간: {total_time:.2f}초")
        
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[Agentic] 예상치 못한 오류: {str(e)} (소요 시간: {elapsed:.2f}초)")
        raise

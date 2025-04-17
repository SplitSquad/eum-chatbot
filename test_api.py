import asyncio
from app.core.llm_client import get_llm_client
from loguru import logger

async def test_api():
    # 경량 모델 테스트
    logger.info("경량 모델 테스트 시작...")
    lightweight_client = get_llm_client(is_lightweight=True)
    lightweight_connected = await lightweight_client.check_connection()
    logger.info(f"경량 모델 연결 상태: {'성공' if lightweight_connected else '실패'}")
    
    if lightweight_connected:
        try:
            response = await lightweight_client.generate("안녕하세요, 테스트입니다.")
            logger.info(f"경량 모델 응답: {response}")
        except Exception as e:
            logger.error(f"경량 모델 생성 오류: {str(e)}")
    
    # 고성능 모델 테스트
    logger.info("\n고성능 모델 테스트 시작...")
    high_performance_client = get_llm_client(is_lightweight=False)
    high_performance_connected = await high_performance_client.check_connection()
    logger.info(f"고성능 모델 연결 상태: {'성공' if high_performance_connected else '실패'}")
    
    if high_performance_connected:
        try:
            response = await high_performance_client.generate("안녕하세요, 테스트입니다.")
            logger.info(f"고성능 모델 응답: {response}")
        except Exception as e:
            logger.error(f"고성능 모델 생성 오류: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_api()) 
from typing import Optional
from loguru import logger
from app.core.llm_client import get_llm_client

class Chatbot:
    """챗봇 클래스"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
    
    async def get_response(self, query: str) -> Optional[str]:
        """
        사용자 질의에 대한 응답을 생성합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            Optional[str]: 생성된 응답
        """
        try:
            prompt = f"""
            사용자의 질문에 대해 적절한 응답을 생성해주세요.
            질문: {query}
            """
            
            response = await self.llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            return None 
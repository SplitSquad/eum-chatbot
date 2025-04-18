from typing import Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client

class Postprocessor:
    """후처리 서비스"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
        logger.info(f"[후처리] 경량 모델 사용: {self.llm_client.model}")
    
    async def postprocess(self, response: str, source_lang: str, rag_type: str) -> Dict[str, Any]:
        """
        응답을 후처리합니다.
        
        Args:
            response: 영어로 된 응답
            source_lang: 원문 언어 코드
            rag_type: 사용된 RAG 유형
            
        Returns:
            Dict[str, Any]: 후처리된 응답
        """
        try:
            logger.info(f"[후처리] 응답 후처리 시작 - 원문 언어: {source_lang}, RAG 유형: {rag_type}")
            
            # 원문 언어로 번역
            if source_lang != 'en':
                prompt = f"""
                Translate the following English text to {source_lang}. 
                Keep the meaning and tone exactly the same.
                Only return the translated text without any additional explanation.
                
                Text to translate:
                {response}
                """
                
                translated_response = await self.llm_client.generate(prompt)
                logger.info(f"[후처리] 번역 완료: {translated_response}")
            else:
                translated_response = response
            
            # RAG 사용 여부 확인
            used_rag = rag_type != "none"
            
            return {
                "response": translated_response,
                "used_rag": used_rag,
                "rag_type": rag_type if used_rag else None
            }
        except Exception as e:
            logger.error(f"후처리 중 오류 발생: {str(e)}")
            return {
                "response": response,  # 번역 실패 시 원문 응답 반환
                "used_rag": False,
                "rag_type": None
            } 
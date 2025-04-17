from typing import Dict, Any
from loguru import logger
from app.core.llm_client import get_llm_client
from app.models.agentic_response import AgentType, ActionType

class AgentClassifier:
    """에이전트 분류기"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """
        에이전트 유형과 필요한 액션을 분류합니다.
        
        Args:
            query: 입력 질의
            
        Returns:
            Dict[str, Any]: 분류 결과
        """
        # 에이전트 유형 분류
        agent_type = await self._classify_agent_type(query)
        
        # 필요한 액션 분류
        action_type = await self._classify_action_type(query)
        
        return {
            "agent_type": agent_type,
            "action_type": action_type
        }
    
    async def _classify_agent_type(self, query: str) -> AgentType:
        """에이전트 유형을 분류합니다."""
        prompt = f"""
        다음 질문을 처리하기 위해 어떤 유형의 에이전트가 필요한지 판단해주세요.
        질문: {query}
        
        다음 중 하나만 답변해주세요:
        - general
        - task
        - domain
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            response = response.strip().lower()
            
            # 응답에서 키워드 추출
            if "task" in response:
                return AgentType.TASK
            elif "domain" in response:
                return AgentType.DOMAIN
            else:
                return AgentType.GENERAL
                
        except Exception as e:
            logger.error(f"에이전트 유형 분류 중 오류 발생: {str(e)}")
            return AgentType.GENERAL
    
    async def _classify_action_type(self, query: str) -> ActionType:
        """필요한 액션 유형을 분류합니다."""
        prompt = f"""
        다음 질문을 처리하기 위해 어떤 유형의 액션이 필요한지 판단해주세요.
        질문: {query}
        
        다음 중 하나만 답변해주세요:
        - inform
        - execute
        - decide
        """
        
        try:
            response = await self.llm_client.generate(prompt)
            response = response.strip().lower()
            
            # 응답에서 키워드 추출
            if "execute" in response:
                return ActionType.EXECUTE
            elif "decide" in response:
                return ActionType.DECIDE
            else:
                return ActionType.INFORM
                
        except Exception as e:
            logger.error(f"액션 유형 분류 중 오류 발생: {str(e)}")
            return ActionType.INFORM 
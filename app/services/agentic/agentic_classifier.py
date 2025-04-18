from typing import Dict, Any, Tuple
from enum import Enum
from loguru import logger
from app.core.llm_client import get_llm_client
from app.models.agentic_response import AgentType, ActionType
from app.config.rag_config import RAGDomain

class AgenticType(str, Enum):
    """에이전틱 기능 유형"""
    GENERAL = "general"  # 일반 대화
    SCHEDULE = "schedule"  # 일정 관리
    TODO = "todo"  # 할 일 관리
    MEMO = "memo"  # 메모 관리
    CALENDAR = "calendar"  # 캘린더 관리
    REMINDER = "reminder"  # 알림 관리

class AgentClassifier:
    """에이전트 분류기"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
        logger.info(f"[분류기] 경량 모델 사용: {self.llm_client.model}")
    
    async def classify(self, query: str) -> Dict[str, Any]:
        """
        에이전트 유형, 액션 유형, 도메인을 분류합니다.
        
        Args:
            query: 입력 질의
            
        Returns:
            Dict[str, Any]: 분류 결과
        """
        logger.info(f"[분류기] 질의 분류 시작: {query}")
        
        # 에이전트 유형 분류
        agent_type = await self._classify_agent_type(query)
        logger.info(f"[분류기] 에이전트 유형: {agent_type.value}")
        
        # 필요한 액션 분류
        action_type = await self._classify_action_type(query)
        logger.info(f"[분류기] 액션 유형: {action_type.value}")
        
        # 도메인 분류
        domain = await self._classify_domain(query)
        logger.info(f"[분류기] 도메인: {domain.value}")
        
        result = {
            "agent_type": agent_type,
            "action_type": action_type,
            "domain": domain
        }
        
        logger.info(f"[분류기] 분류 완료: {result}")
        return result
    
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
            logger.info(f"[분류기] 에이전트 유형 분류 시작")
            response = await self.llm_client.generate(prompt)
            response = response.strip().lower()
            logger.info(f"[분류기] 에이전트 유형 분류 결과: {response}")
            
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
            logger.info(f"[분류기] 액션 유형 분류 시작")
            response = await self.llm_client.generate(prompt)
            response = response.strip().lower()
            logger.info(f"[분류기] 액션 유형 분류 결과: {response}")
            
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
    
    async def _classify_domain(self, query: str) -> RAGDomain:
        """도메인을 분류합니다."""
        prompt = f"""
        다음 질문이 어떤 도메인에 속하는지 판단해주세요.
        질문: {query}
        
        다음 중 하나만 답변해주세요:
        - visa_law: 비자/법률 관련 질문
        - social_security: 사회보장제도 관련 질문
        - tax_finance: 세금/금융 관련 질문
        - medical_health: 의료/건강 관련 질문
        - employment: 취업 관련 질문
        - daily_life: 일상생활 관련 질문
        """
        
        try:
            logger.info(f"[분류기] 도메인 분류 시작")
            response = await self.llm_client.generate(prompt)
            response = response.strip().lower()
            logger.info(f"[분류기] 도메인 분류 결과: {response}")
            
            # 응답에서 도메인 추출
            for domain in RAGDomain:
                if domain.value in response:
                    return domain
            
            # 기본값으로 일상생활 도메인 반환
            return RAGDomain.DAILY_LIFE
                
        except Exception as e:
            logger.error(f"도메인 분류 중 오류 발생: {str(e)}")
            return RAGDomain.DAILY_LIFE

class AgenticClassifier:
    """에이전틱 분류기"""
    
    def __init__(self):
        self.llm_client = get_llm_client(is_lightweight=True)
        logger.info(f"[에이전틱 분류] 경량 모델 사용: {self.llm_client.model}")
    
    async def classify(self, query: str) -> AgenticType:
        """
        질의를 분류합니다.
        
        Args:
            query: 사용자 질의
            
        Returns:
            AgenticType: 에이전틱 기능 유형
        """
        try:
            logger.info(f"[에이전틱 분류] 질의 분류 시작: {query}")
            
            # 질의 유형 분류
            agentic_type = await self._classify_agentic_type(query)
            logger.info(f"[에이전틱 분류] 기능 유형: {agentic_type.value}")
            
            return agentic_type
            
        except Exception as e:
            logger.error(f"분류 중 오류 발생: {str(e)}")
            return AgenticType.GENERAL
    
    async def _classify_agentic_type(self, query: str) -> AgenticType:
        """에이전틱 기능 유형을 분류합니다."""
        try:
            # TODO: LLM을 활용한 기능 유형 분류 구현
            # 임시로 모든 질의를 일반 대화로 분류
            return AgenticType.GENERAL
        except Exception as e:
            logger.error(f"기능 유형 분류 중 오류 발생: {str(e)}")
            return AgenticType.GENERAL 
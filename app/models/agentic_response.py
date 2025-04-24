from enum import Enum
from pydantic import BaseModel


# TODO: 에이전트 응답 타입 정의
# 현재는 빈 파일로 두고, 추후 에이전트 관련 응답 타입을 정의할 예정 

class AgentType(str, Enum):
    """에이전트 유형"""
    GENERAL = "general"
    TASK = "task"
    DOMAIN = "domain"

class ActionType(str, Enum):
    """액션 유형"""
    INFORM = "inform"
    EXECUTE = "execute"
    DECIDE = "decide"

class ResumeInput(BaseModel):
        name: str
        job_title: str
        experience: str
        skills: str
        education: str

class ResumeResponse(BaseModel):
        pdf_path: str  # 생성된 PDF 파일 경로
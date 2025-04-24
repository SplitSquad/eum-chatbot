from typing import Dict, Any, Optional, List
from enum import Enum
from loguru import logger
from app.services.common.preprocessor import translate_query
from app.services.common.postprocessor import Postprocessor
from app.services.agentic.agentic_classifier import AgenticClassifier, AgenticType
from app.services.agentic.agentic_response_generator import AgenticResponseGenerator
from app.services.chatbot.chatbot_classifier import RAGType


class AgentState(str, Enum):
    """에이전트 상태"""
    GENERAL = "일반"  # 일반 상태 (기본)
    SCHEDULE = "일정 변경"  # 일정 변경 기능
    JOB = "구직 도움"  # 구직 도움 기능
    WRITING = "글 작성"  # 글 작성 기능


class Agentic:
    """에이전트 클래스 - 워크플로우 관리"""
    
    def __init__(self):
        self.classifier = AgenticClassifier()
        self.response_generator = AgenticResponseGenerator()
        self.postprocessor = Postprocessor()
        # 사용자별 상태 저장 딕셔너리
        self.user_states: Dict[str, Dict[str, Any]] = {}
        logger.info("[에이전트] 초기화 완료")
    
    def _get_user_state(self, uid: str) -> Dict[str, Any]:
        """사용자 상태 정보를 가져옵니다."""
        if uid not in self.user_states:
            # 새 사용자일 경우 기본 상태 설정
            self.user_states[uid] = {
                "state": AgentState.GENERAL,  # 기본 상태는 '일반'
                "context": {},  # 상태별 컨텍스트 정보
                "collected_info": {},  # 수집된 정보
            }
        return self.user_states[uid]
    
    def _update_user_state(self, uid: str, state: AgentState, context: Dict[str, Any] = None) -> None:
        """사용자 상태를 업데이트합니다."""
        user_state = self._get_user_state(uid)
        user_state["state"] = state
        if context:
            user_state["context"].update(context)
        logger.info(f"[에이전트] 사용자 {uid}의 상태가 '{state.value}'로 변경되었습니다.")
    
    def _reset_user_state(self, uid: str) -> None:
        """사용자 상태를 초기화합니다."""
        self.user_states[uid] = {
            "state": AgentState.GENERAL,
            "context": {},
            "collected_info": {},
        }
        logger.info(f"[에이전트] 사용자 {uid}의 상태가 초기화되었습니다.")
    
    def _check_cancel_intent(self, query: str) -> bool:
        """사용자 질의에서 중단 의도를 확인합니다."""
        cancel_keywords = ["그만", "중단", "취소", "취소할래", "그만할래", "이 기능 사용 안할래", "안할래"]
        return any(keyword in query for keyword in cancel_keywords)
    
    async def _classify_agent_functionality(self, query: str) -> AgentState:
        """질의를 분석하여 에이전트 기능을 분류합니다."""
        # 간단한 규칙 기반 분류 (실제로는 LLM 기반 분류를 구현해야 함)
        if "일정" in query or "스케줄" in query or "캘린더" in query or "약속" in query:
            return AgentState.SCHEDULE
        elif "구직" in query or "취업" in query or "일자리" in query or "직장" in query:
            return AgentState.JOB
        elif "글" in query or "작성" in query or "메일" in query or "편지" in query:
            return AgentState.WRITING
        else:
            return AgentState.GENERAL
    
    async def _process_schedule_state(self, query: str, uid: str) -> Dict[str, Any]:
        """일정 변경 기능 상태 처리"""
        user_state = self._get_user_state(uid)
        context = user_state["context"]
        collected_info = user_state["collected_info"]
        
        # 기능 중단 의도 확인
        if self._check_cancel_intent(query):
            self._reset_user_state(uid)
            return {
                "response": "일정 변경 기능을 중단했습니다. 다른 도움이 필요하신가요?",
                "metadata": {
                    "state": AgentState.GENERAL.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 필요한 정보 수집
        if "date" not in collected_info:
            # 날짜 정보 요청
            collected_info["date"] = query if "년" in query or "월" in query or "일" in query else None
            if not collected_info["date"]:
                return {
                    "response": "일정을 변경할 날짜를 알려주세요 (예: 2025년 5월 10일)",
                    "metadata": {
                        "state": AgentState.SCHEDULE.value,
                        "query": query,
                        "uid": uid
                    }
                }
        
        if "time" not in collected_info:
            # 시간 정보 요청
            collected_info["time"] = query if "시" in query or "분" in query else None
            if not collected_info["time"]:
                return {
                    "response": "일정 시간을 알려주세요 (예: 오후 2시 30분)",
                    "metadata": {
                        "state": AgentState.SCHEDULE.value,
                        "query": query,
                        "uid": uid
                    }
                }
        
        if "title" not in collected_info:
            # 일정 제목 요청
            collected_info["title"] = query
            
        # 모든 정보가 수집되면 일정 변경 수행
        # TODO: 실제 일정 변경 로직 구현
        
        # 기능 완료 후 상태 초기화
        self._reset_user_state(uid)
        
        return {
            "response": f"{collected_info['date']} {collected_info['time']}에 '{collected_info['title']}' 일정이 추가되었습니다.",
            "metadata": {
                "state": AgentState.GENERAL.value,
                "query": query,
                "uid": uid,
                "completed": True
            }
        }
    
    async def _process_job_state(self, query: str, uid: str) -> Dict[str, Any]:
        """구직 도움 기능 상태 처리"""
        user_state = self._get_user_state(uid)
        context = user_state["context"]
        collected_info = user_state["collected_info"]
        
        # 기능 중단 의도 확인
        if self._check_cancel_intent(query):
            self._reset_user_state(uid)
            return {
                "response": "구직 도움 기능을 중단했습니다. 다른 도움이 필요하신가요?",
                "metadata": {
                    "state": AgentState.GENERAL.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 필요한 정보 수집
        if "field" not in collected_info:
            # 직무 분야 요청
            collected_info["field"] = query
            return {
                "response": "어떤 직무나 업계에서 일하고 싶으신가요?",
                "metadata": {
                    "state": AgentState.JOB.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        if "experience" not in collected_info:
            # 경력 정보 요청
            collected_info["experience"] = query
            return {
                "response": "관련 경력이나 스킬을 알려주세요.",
                "metadata": {
                    "state": AgentState.JOB.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 모든 정보가 수집되면 구직 도움 수행
        # TODO: 실제 구직 도움 로직 구현
        
        # 기능 완료 후 상태 초기화
        self._reset_user_state(uid)
        
        return {
            "response": f"{collected_info['field']} 분야의 구직 정보를 찾아보았습니다. {collected_info['experience']}와 같은 경험을 가진 분에게 추천하는 직무는...",
            "metadata": {
                "state": AgentState.GENERAL.value,
                "query": query,
                "uid": uid,
                "completed": True
            }
        }
    
    async def _process_writing_state(self, query: str, uid: str) -> Dict[str, Any]:
        """글 작성 기능 상태 처리"""
        user_state = self._get_user_state(uid)
        context = user_state["context"]
        collected_info = user_state["collected_info"]
        
        # 기능 중단 의도 확인
        if self._check_cancel_intent(query):
            self._reset_user_state(uid)
            return {
                "response": "글 작성 기능을 중단했습니다. 다른 도움이 필요하신가요?",
                "metadata": {
                    "state": AgentState.GENERAL.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 필요한 정보 수집
        if "type" not in collected_info:
            # 글 유형 요청
            collected_info["type"] = query
            return {
                "response": "어떤 종류의 글을 작성하고 싶으신가요? (예: 이메일, 보고서, 편지 등)",
                "metadata": {
                    "state": AgentState.WRITING.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        if "topic" not in collected_info:
            # 주제 요청
            collected_info["topic"] = query
            return {
                "response": "글의 주제나 내용을 간략히 알려주세요.",
                "metadata": {
                    "state": AgentState.WRITING.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        if "tone" not in collected_info:
            # 어조 요청
            collected_info["tone"] = query
            return {
                "response": "어떤 어조로 작성할까요? (예: 공식적, 친근한, 전문적 등)",
                "metadata": {
                    "state": AgentState.WRITING.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 모든 정보가 수집되면 글 작성 수행
        # TODO: 실제 글 작성 로직 구현
        
        # 기능 완료 후 상태 초기화
        self._reset_user_state(uid)
        
        return {
            "response": f"{collected_info['type']} 형식으로 '{collected_info['topic']}' 주제에 대해 {collected_info['tone']} 어조로 작성한 글입니다:\n\n[작성된 글 내용]",
            "metadata": {
                "state": AgentState.GENERAL.value,
                "query": query,
                "uid": uid,
                "completed": True
            }
        }
    
    async def get_response(self, query: str, uid: str) -> Dict[str, Any]:
        """질의에 대한 응답을 생성합니다."""
        try:
            logger.info(f"[WORKFLOW] ====== Starting agentic workflow for user {uid} ======")
            logger.info(f"[WORKFLOW] Original query: {query}")
            
            # 현재 사용자 상태 가져오기
            user_state = self._get_user_state(uid)
            current_state = user_state["state"]
            logger.info(f"[에이전트] 현재 사용자 상태: {current_state.value}")
            
            # 중단 의도 확인 (일반 상태가 아닐 때)
            if current_state != AgentState.GENERAL and self._check_cancel_intent(query):
                self._reset_user_state(uid)
                return {
                    "response": f"{current_state.value} 기능을 중단했습니다. 다른 도움이 필요하신가요?",
                    "metadata": {
                        "state": AgentState.GENERAL.value,
                        "query": query,
                        "uid": uid
                    }
                }
            
            # 1. 전처리 (언어 감지 및 번역)
            logger.info(f"[WORKFLOW] Step 1: Preprocessing (language detection and translation)")
            translation_result = await translate_query(query)
            source_lang = translation_result["lang_code"]
            english_query = translation_result["translated_query"]
            logger.info(f"[에이전트] 언어 감지 완료 - 소스 언어: {source_lang}, 영어 번역: {english_query}")
            
            # 2. 상태에 따른 분기 처리
            response_data = None
            
            # 2.1 일반 상태일 경우 - 기능 분류 수행
            if current_state == AgentState.GENERAL:
                logger.info(f"[WORKFLOW] Step 2: Classification in GENERAL state")
                
                # 분류 수행
                agent_state = await self._classify_agent_functionality(english_query)
                logger.info(f"[에이전트] 기능 분류 결과: {agent_state.value}")
                
                # 분류 결과에 따라 상태 업데이트 및 처리
                if agent_state == AgentState.SCHEDULE:
                    self._update_user_state(uid, AgentState.SCHEDULE)
                    response_data = {
                        "response": "일정 변경 기능을 시작합니다. 어떤 일정을 추가하거나 변경하고 싶으신가요?",
                        "metadata": {
                            "state": AgentState.SCHEDULE.value,
                            "query": query,
                            "uid": uid
                        }
                    }
                elif agent_state == AgentState.JOB:
                    self._update_user_state(uid, AgentState.JOB)
                    response_data = {
                        "response": "구직 도움 기능을 시작합니다. 어떤 직무에 관심이 있으신가요?",
                        "metadata": {
                            "state": AgentState.JOB.value,
                            "query": query,
                            "uid": uid
                        }
                    }
                elif agent_state == AgentState.WRITING:
                    self._update_user_state(uid, AgentState.WRITING)
                    response_data = {
                        "response": "글 작성 기능을 시작합니다. 어떤 종류의 글을 작성하고 싶으신가요?",
                        "metadata": {
                            "state": AgentState.WRITING.value,
                            "query": query,
                            "uid": uid
                        }
                    }
                else:
                    # 일반 질의 처리 (기존 챗봇 로직 활용)
                    agentic_type = await self.classifier.classify(english_query)
                    logger.info(f"[에이전트] 에이전틱 유형: {agentic_type.value}")
                    
                    # 응답 생성
                    result = await self.response_generator.generate_response(english_query, agentic_type)
                    response_data = {
                        "response": result["response"],
                        "metadata": {
                            "state": AgentState.GENERAL.value,
                            "query": query,
                            "english_query": english_query,
                            "uid": uid,
                            "agentic_type": agentic_type.value
                        }
                    }
            
            # 2.2 특정 기능 상태일 경우 - 해당 기능 처리 수행
            elif current_state == AgentState.SCHEDULE:
                logger.info(f"[WORKFLOW] Step 2: Processing in SCHEDULE state")
                response_data = await self._process_schedule_state(english_query, uid)
            
            elif current_state == AgentState.JOB:
                logger.info(f"[WORKFLOW] Step 2: Processing in JOB state")
                response_data = await self._process_job_state(english_query, uid)
            
            elif current_state == AgentState.WRITING:
                logger.info(f"[WORKFLOW] Step 2: Processing in WRITING state")
                response_data = await self._process_writing_state(english_query, uid)
            
            # 3. 후처리 (원문 언어로 번역)
            logger.info(f"[WORKFLOW] Step 3: Postprocessing (translation back to original language)")
            if source_lang != "en":
                processed_response = await self.postprocessor.postprocess(response_data["response"], source_lang, "general")
                response_data["response"] = processed_response["response"]
                response_data["metadata"]["translated"] = True
            
            logger.info(f"[WORKFLOW] ====== Agentic workflow completed for user {uid} ======")
            return response_data
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {str(e)}")
            logger.error(f"[WORKFLOW] ====== Error in agentic workflow: {str(e)} ======")
            return {
                "response": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "metadata": {
                    "query": query,
                    "state": "error",
                    "uid": uid,
                    "error": str(e)
                }
            } 
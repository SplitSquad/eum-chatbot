from googleapiclient.discovery import build # Google API 클라이언트 생성 도구
from google_auth_oauthlib.flow import InstalledAppFlow # OAuth 인증 흐름을 다루는 도구
from google.auth.transport.requests import Request # 토큰 갱신 시 필요한 요청 객체
from datetime import datetime #날짜 다룰 때 사용
# 파일 저장 및 불러오기용 (토큰 저장)
import os
import pickle
import requests
import json
import re
from enum import Enum
from typing import Dict, Any, List, Optional
from loguru import logger
from dotenv import load_dotenv
load_dotenv(".env")  # 프로젝트 루트의 .env 파일 로딩
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


class CalendarState(str, Enum):
    """일정 관리 상태"""
    INITIAL = "initial"  # 초기 상태
    ADD = "add"  # 일정 추가
    DELETE = "delete"  # 일정 삭제
    EDIT = "edit"  # 일정 수정
    DATE_INPUT = "date_input"  # 날짜 입력
    TIME_INPUT = "time_input"  # 시간 입력
    TITLE_INPUT = "title_input"  # 제목 입력
    CONFIRM = "confirm"  # 확인


class AgenticCalendar:
    """일정 관리 에이전트"""
    
    def __init__(self):
        """일정 관리 에이전트 초기화"""
        logger.info("[일정 관리] 에이전트 초기화")
        
    async def process_query(self, query: str, uid: str, user_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        일정 관리 관련 질의 처리
        
        Args:
            query: 사용자 질의
            uid: 사용자 ID
            user_states: 사용자별 상태 정보
            
        Returns:
            Dict[str, Any]: 응답 데이터
        """
        # 사용자 상태 확인
        if uid not in user_states:
            user_states[uid] = {
                "state": "general",
                "context": {},
                "calendar": {
                    "state": CalendarState.INITIAL,
                    "collected_info": {}
                }
            }
            
        calendar_state = user_states[uid]["calendar"]["state"]
        collected_info = user_states[uid]["calendar"]["collected_info"]
        
        logger.info(f"[일정 관리] 현재 상태: {calendar_state}")
        
        # 초기 상태일 경우 작업 분류
        if calendar_state == CalendarState.INITIAL:
            operation = Input_analysis(query)
            user_states[uid]["calendar"]["operation"] = operation
            
            if operation == "add":
                user_states[uid]["calendar"]["state"] = CalendarState.DATE_INPUT
                return {
                    "response": "일정을 추가합니다. 날짜를 알려주세요 (예: 2025년 5월 10일)",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.DATE_INPUT.value,
                        "query": query,
                        "uid": uid
                    }
                }
            elif operation == "delete":
                # 일정 목록 표시 후 삭제할 일정 선택
                Calendar_list()
                user_states[uid]["calendar"]["state"] = CalendarState.DELETE
                return {
                    "response": "삭제할 일정을 알려주세요.",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.DELETE.value,
                        "query": query,
                        "uid": uid
                    }
                }
            elif operation == "edit":
                # 일정 목록 표시 후 수정할 일정 선택
                Calendar_list()
                user_states[uid]["calendar"]["state"] = CalendarState.EDIT
                return {
                    "response": "수정할 일정을 알려주세요.",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.EDIT.value,
                        "query": query,
                        "uid": uid
                    }
                }
            else:
                # 분류할 수 없는 경우
                return {
                    "response": "죄송합니다. 일정 명령을 이해하지 못했습니다. 일정 추가, 삭제, 또는 수정을 원하시나요?",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.INITIAL.value,
                        "query": query,
                        "uid": uid
                    }
                }
        
        # 일정 추가 프로세스
        elif calendar_state == CalendarState.DATE_INPUT:
            # 날짜 정보 수집
            collected_info["date"] = query if "년" in query or "월" in query or "일" in query else None
            if not collected_info["date"]:
                return {
                    "response": "날짜 형식이 맞지 않습니다. 예를 들어 '2025년 5월 10일'과 같이 입력해주세요.",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.DATE_INPUT.value,
                        "query": query,
                        "uid": uid
                    }
                }
            
            # 다음 단계로 진행
            user_states[uid]["calendar"]["state"] = CalendarState.TIME_INPUT
            return {
                "response": "시간을 알려주세요 (예: 오후 2시 30분)",
                "metadata": {
                    "state": "calendar",
                    "calendar_state": CalendarState.TIME_INPUT.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        elif calendar_state == CalendarState.TIME_INPUT:
            # 시간 정보 수집
            collected_info["time"] = query if "시" in query or "분" in query else None
            if not collected_info["time"]:
                return {
                    "response": "시간 형식이 맞지 않습니다. 예를 들어 '오후 2시 30분'과 같이 입력해주세요.",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.TIME_INPUT.value,
                        "query": query,
                        "uid": uid
                    }
                }
            
            # 다음 단계로 진행
            user_states[uid]["calendar"]["state"] = CalendarState.TITLE_INPUT
            return {
                "response": "일정 제목을 알려주세요.",
                "metadata": {
                    "state": "calendar",
                    "calendar_state": CalendarState.TITLE_INPUT.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        elif calendar_state == CalendarState.TITLE_INPUT:
            # 제목 정보 수집
            collected_info["title"] = query
            
            # 일정 추가 준비 및 확인 요청
            user_states[uid]["calendar"]["state"] = CalendarState.CONFIRM
            
            datetime_str = f"{collected_info['date']} {collected_info['time']}"
            return {
                "response": f"'{collected_info['title']}' 일정을 {datetime_str}에 추가하시겠습니까? (예/아니오)",
                "metadata": {
                    "state": "calendar",
                    "calendar_state": CalendarState.CONFIRM.value,
                    "query": query,
                    "uid": uid
                }
            }
        
        elif calendar_state == CalendarState.CONFIRM:
            # 확인 응답 처리
            if "예" in query or "네" in query or "좋아" in query or "추가" in query:
                # 일정 추가 실행
                schedule_input = f"{collected_info['date']} {collected_info['time']}에 {collected_info['title']} 일정"
                make_event = MakeSchedule(schedule_input)
                add_event(make_event)
                
                # 상태 초기화
                user_states[uid]["calendar"]["state"] = CalendarState.INITIAL
                user_states[uid]["calendar"]["collected_info"] = {}
                
                return {
                    "response": f"{collected_info['date']} {collected_info['time']}에 '{collected_info['title']}' 일정이 추가되었습니다.",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.INITIAL.value,
                        "completed": True,
                        "query": query,
                        "uid": uid
                    }
                }
            else:
                # 취소
                user_states[uid]["calendar"]["state"] = CalendarState.INITIAL
                user_states[uid]["calendar"]["collected_info"] = {}
                
                return {
                    "response": "일정 추가가 취소되었습니다.",
                    "metadata": {
                        "state": "calendar",
                        "calendar_state": CalendarState.INITIAL.value,
                        "query": query,
                        "uid": uid
                    }
                }
        
        # 일정 삭제 프로세스
        elif calendar_state == CalendarState.DELETE:
            delete_event(query)
            
            # 상태 초기화
            user_states[uid]["calendar"]["state"] = CalendarState.INITIAL
            
            return {
                "response": "일정이 삭제되었습니다.",
                "metadata": {
                    "state": "calendar",
                    "calendar_state": CalendarState.INITIAL.value,
                    "completed": True,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 일정 수정 프로세스
        elif calendar_state == CalendarState.EDIT:
            edit_event(query)
            
            # 상태 초기화
            user_states[uid]["calendar"]["state"] = CalendarState.INITIAL
            
            return {
                "response": "일정이 수정되었습니다.",
                "metadata": {
                    "state": "calendar",
                    "calendar_state": CalendarState.INITIAL.value,
                    "completed": True,
                    "query": query,
                    "uid": uid
                }
            }
        
        # 예상치 못한 상태
        return {
            "response": "죄송합니다. 일정 처리 중 오류가 발생했습니다.",
            "metadata": {
                "state": "calendar",
                "error": "Unknown calendar state",
                "query": query,
                "uid": uid
            }
        }

################################################ 캘린더 일정 리스트로 반환
def Calendar_list():
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    # 일정 목록 가져오기
    events_result = service.events().list(
        calendarId='primary',
        maxResults=2500,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    events = events_result.get('items', [])

    if not events:
        print("📭 등록된 일정이 없습니다.")
        return


    # 요약용 이벤트 리스트 (AI가 분석할 수 있도록 최소 정보만 포함)
    simplified_events = [
        {
            "id": event["id"],
            "summary": event.get("summary", "(제목 없음)"),
            "start": event['start'].get('dateTime', event['start'].get('date')),
            "end": event['end'].get('dateTime', event['end'].get('date'))
        }
        for event in events
    ]
    def format_event_pretty(event: dict) -> str:
        return (
            f'"id": "{event["id"]}",\n'
            f'"summary": "{event["summary"]}",\n'
            f'"start": "{event["start"]}",\n'
            f'"end": "{event["end"]}"'
        )
    # 예시 사용법
    events = [
        {
            "id": "########",
            "summary": "점심약속",
            "start": "2025-04-22T12:30:00+09:00",
            "end": "2025-04-22T14:00:00+09:00"
        }
    ]

    # 포맷된 문자열을 저장할 리스트
    formatted_events = [format_event_pretty(event) for event in simplified_events]

    # 결과 출력 (선택)
    for formatted in formatted_events:
        print("------")
        print(formatted)

    return formatted_events
################################################ 캘린더 일정 리스트로 반환

################################################ 구글 켈린더 엑세스
# 인증 범위 지정
SCOPES = ['https://www.googleapis.com/auth/calendar']

# 사용자 인증 + access_token 관리
def get_credentials():
    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle','rb') as token : 
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES )
            creds = flow.run_local_server(port=8080)

        # 새 토큰 저장
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds  # ✅ 함수 끝에서 항상 반환
################################################ 구글 켈린더 엑세스

################################################ user input 분류
def Input_analysis(user_input):
    
    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
    )
    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "input": {"type": "string"},
            "output": {"type": "string"},
        }
    })
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You're a scheduling assistant.

    Classify the user's sentence into one of the following three categories:
    - Add (new schedule)
    - Delete (remove schedule)
    - Edit (modify schedule)

    Return the result as JSON like:
    {{
    "input": "...",
    "output": "add"  // or "delete" or "edit"
    }}

    Examples:
    # ✅ ADD
    {{"input": "오늘 오후에 영화 보자", "output": "add"}},
    {{"input": "5월 3일에 생일 파티 일정 추가해줘", "output": "add"}},
    {{"input": "이번 주 금요일에 미용실 예약 좀 넣어줘", "output": "add"}},
    {{"input": "4월 20일에 친구랑 저녁 약속 있어", "output": "add"}},
    {{"input": "모레 점심 먹자", "output": "add"}},
    {{"input": "내일 오전 9시에 회의 있어", "output": "add"}},
    {{"input": "주말에 등산 일정 잡아줘", "output": "add"}},
    {{"input": "다음주 화요일에 프로젝트 발표 있어", "output": "add"}},
    {{"input": "오늘 밤에 헬스장 갈 거야", "output": "add"}},
    {{"input": "7시에 엄마랑 전화하기 일정 넣어줘", "output": "add"}},

    # ✅ DELETE
    {{"input": "오늘 저녁 약속 취소해줘", "output": "delete"}},
    {{"input": "5시 회의 일정 없애줘", "output": "delete"}},
    {{"input": "내일 생일 파티 취소됐어", "output": "delete"}},
    {{"input": "친구 만나는 일정 지워줘", "output": "delete"}},
    {{"input": "방금 넣은 일정 삭제해줘", "output": "delete"}},
    {{"input": "이번 주말 일정 취소할래", "output": "delete"}},
    {{"input": "3시에 예약한 거 없애줘", "output": "delete"}},
    {{"input": "다음주 월요일 약속 취소해줘", "output": "delete"}},
    {{"input": "쇼핑 일정 삭제해줘", "output": "delete"}},
    {{"input": "헬스장 안 가기로 했어", "output": "delete"}},

    # ✅ EDIT
    {{"input": "내일 회의 시간 바꿔줘", "output": "edit"}},
    {{"input": "오늘 약속 3시로 변경해줘", "output": "edit"}},
    {{"input": "저녁 6시 약속 7시로 옮겨줘", "output": "edit"}},
    {{"input": "생일 파티 장소 바뀌었어", "output": "edit"}},
    {{"input": "오후 회의 Zoom 링크로 수정해줘", "output": "edit"}},
    {{"input": "영화 시간 2시로 바꿔줘", "output": "edit"}},
    {{"input": "점심 시간 다시 조정해줘", "output": "edit"}},
    {{"input": "오늘 일정 제목 바꿔줘", "output": "edit"}},
    {{"input": "내일 약속 위치 바뀜", "output": "edit"}},
    {{"input": "저녁 약속 시간 변경해줘", "output": "edit"}}
    """),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print(json.dumps(result, indent=2))

        return json.dumps(result, indent=2)
        
    description = user_input
    response = parse_product(description)
    print("response\n",response)

    response = json.loads(response)  # 문자열 → 딕셔너리
    print("response output\n",response["output"])

    return response["output"]
################################################ user input 분류

################################################ 일정 추가
from langchain.output_parsers import StructuredOutputParser
from pydantic import BaseModel, Field

# 현재 날짜와 시간
now = datetime.now()
def MakeSchedule(user_input):

    llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.environ.get("GROQ_API_KEY")
    )
    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "location": {"type": "string"},
            "description": { "type": "string"},
            "start":{
                "dateTime": {"type": "string"},
                "timeZone": {"type": "string"}
            },
            "end":{
                "dateTime": {"type": "string"},
                "timeZone": {"type": "string"}
            },
            "reminders":{
                "useDefault": {"type": "boolean"},
                "overrides": {"type": "string"}
            }
        }
    })
    system_prompt = f"""
    0. Always remember the date : now
    1. Please write as in the Examples:
    {{{{  
    "summary": "회의",
    "location": "강남 사무실",
    "description": "내일 오전 팀 회의",
    "start": {{{{  
        "dateTime": "2025-04-18T10:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "end": {{{{
        "dateTime": "2025-04-18T11:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "reminders": {{{{
        "useDefault": false,
        "overrides": [
        {{{{ "method": "popup", "minutes": 10 }}}}
        ]
    }}}}
    }}}},

    {{{{  
    "summary": "저녁 약속",
    "location": "홍대 맛집",
    "description": "주말 친구와 저녁 식사",
    "start": {{{{  
        "dateTime": "2025-04-20T18:30:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "end": {{{{
        "dateTime": "2025-04-20T20:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "reminders": {{{{
        "useDefault": false,
        "overrides": [
        {{{{ "method": "popup", "minutes": 10 }}}}
        ]
    }}}}
    }}}},

    {{{{  
    "summary": "운동",
    "location": "피트니스 센터",
    "description": "저녁 헬스장 운동",
    "start": {{{{  
        "dateTime": "2025-04-18T19:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "end": {{{{
        "dateTime": "2025-04-18T20:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "reminders": {{{{
        "useDefault": false,
        "overrides": [
        {{{{ "method": "popup", "minutes": 10 }}}}
        ]
    }}}}
    }}}},

    {{{{  
    "summary": "회의 준비",
    "location": "재택",
    "description": "프로젝트 발표 자료 준비",
    "start": {{{{  
        "dateTime": "2025-04-19T09:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "end": {{{{
        "dateTime": "2025-04-19T11:00:00+09:00",
        "timeZone": "Asia/Seoul"
    }}}},
    "reminders": {{{{
        "useDefault": false,
        "overrides": [
        {{{{ "method": "popup", "minutes": 10 }}}}
        ]
    }}}}
    }}}}
    ...
    ⚠️ Do NOT include any explanation or message. ONLY return a valid JSON object. No natural language.

    """

    prompt=system_prompt
    print("prompt\n",prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print("description\n",description)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    description = user_input

    response = parse_product(description)
    print("response\n",response)

    return response

# 실제 이벤트 등록 함수
# 위에서 얻은 인증 정보로 API를 사용할 수 있는 service 객체 생성
def add_event(make_event):
    try:
        creds = get_credentials()
        service = build('calendar', 'v3', credentials=creds)
        event = make_event
        print("\n📤 보내는 이벤트 JSON:")
        print(json.dumps(event, indent=4, ensure_ascii=False))
        event_result = service.events().insert(
            calendarId='primary',
            body=event
        ).execute()

        print(f"\n✅ 일정이 추가되었습니다: {event_result.get('htmlLink')}")
    except Exception as e:
        print("❌ 일정 추가 중 오류 발생:", e)
################################################ 일정 추가

################################################ 일정 삭제
def delete_event(user_input):

    formatted_events=Calendar_list()

    # 결과 출력 (선택)
    for formatted in formatted_events:
        print("------")
        print(formatted)

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "summary": {"type": "string"},
            "start": {"type": "string"},
            "end" : {"type": "string"}
        }
    })

    example_output = '{{"id": "########", "summary": "점심약속", "start": "2025-04-22T12:30:00+09:00", "end": "2025-04-22T14:00:00+09:00"}}'

    # 프롬프트 문자열 구성
    system_prompt_template = f"""
    1. Please find the schedule you want to cancel
    2. It's a schedule: {formatted_events}
    3. This is an example output

    "output": 
    "id": "########", 
    "summary": "점심약속", 
    "start": "2025-04-22T12:30:00+09:00", 
    "end": "2025-04-22T14:00:00+09:00"
    
    ⚠️ Do NOT include any explanation or message. ONLY return a valid JSON object. No natural language.
    """
    
    print("system_prompt_template\n",system_prompt_template)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template ),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print(json.dumps(result, indent=2))
        return json.dumps(result, indent=2)

    description = user_input

    response = parse_product(description)
    print("response\n",response)

    response_dict = json.loads(response)
    print("response['id']\n",response_dict['id'])
    delete_event_by_id(response_dict['id'])
   

# ✅ 주어진 event_id로 일정 삭제
def delete_event_by_id(event_id):
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    try:
        service.events().delete(calendarId='primary', eventId=event_id).execute()
        print(f"🗑️ 일정이 성공적으로 삭제되었습니다: {event_id}")
    except Exception as e:
        print(f"❌ 삭제 실패: {e}")
################################################ 일정 삭제

################################################ 일정 수정
def edit_event(user_input):

    formatted_events=Calendar_list()

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    parser = JsonOutputParser(pydantic_object={
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "summary": {"type": "string"},
            "start": {"type": "string"},
            "end" : {"type": "string"}
        }
    })

    # 프롬프트 문자열 구성
    system_prompt_template = f"""
    1. I would like to ask you to change the schedule.
    2. It's a schedule: {formatted_events}
    3. This is an example output

    "output": 
    "id": "########",
    "summary": "점심약속",
    "start": "2025-04-22T12:30:00+09:00",
    "end": "2025-04-22T14:00:00+09:00"

    ⚠️ Do NOT include any explanation or message. ONLY return a valid JSON object. No natural language.
    """
    
    print("system_prompt_template\n",system_prompt_template)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template ),
        ("user", "{input}")
    ])

    chain = prompt | llm | parser

    def parse_product(description: str) -> dict:
        result = chain.invoke({"input": description})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return json.dumps(result, indent=2, ensure_ascii=False)

    description = user_input

    response = parse_product(description)
    print("response\n",response)

    response_dict = json.loads(response)
    print("response_dict",response_dict)

    event_id = response_dict["id"]
    updated_fields = {
        "summary": response_dict["summary"],
        "start": {
            "dateTime": response_dict["start"]
        },
        "end": {
            "dateTime": response_dict["end"]
        }
    }
    
    update_event_by_id(event_id,updated_fields)
   

# ✅ 주어진 event_id로 일정 수정
def update_event_by_id(event_id, updated_fields: dict):
    creds = get_credentials()
    service = build('calendar', 'v3', credentials=creds)

    try:
        # 기존 이벤트 정보 가져오기
        event = service.events().get(calendarId='primary', eventId=event_id).execute()

        # 수정할 필드 반영
        event.update(updated_fields)

        # 이벤트 업데이트
        updated_event = service.events().update(
            calendarId='primary',
            eventId=event_id,
            body=event
        ).execute()

        print("✅ 일정이 성공적으로 수정되었습니다:")
        print(f"- 제목: {updated_event.get('summary')}")
        print(f"- 시작: {updated_event['start'].get('dateTime')}")
        print(f"- 종료: {updated_event['end'].get('dateTime')}")
    except Exception as e:
        print(f"❌ 수정 실패: {e}")
################################################ 일정 수정

# 실행 진입전
if __name__ == '__main__':

    user_input = input("🗓️ 일정을 입력하세요: ")

    classification = Input_analysis(user_input)
    print("classification",classification)
    
    if classification == "add" :
        print("일정 추가")        
        make_event = MakeSchedule(user_input) ## 이벤트 생성
        add_event( make_event ) ## 이벤트 추가
    elif classification == "edit" : 
        print("일정 수정")
        edit_event(user_input)    
    elif classification == "delete" : 
        print("일정 삭제")
        delete_event(user_input)
    else : 
        print('알 수 없는 명령입니다.')

    


# EUM Chatbot

이음 챗봇은 외국인을 위한 종합 정보 제공 챗봇 서비스입니다. 다양한 도메인의 전문 지식을 제공하며, RAG(Retrieval-Augmented Generation) 기술을 활용하여 정확하고 신뢰할 수 있는 정보를 제공합니다.

## 주요 기능

- 다국어 지원 (한국어, 영어 등)
- 도메인별 전문 지식 제공
  - 비자/법률 (VISA_LAW)
  - 사회보장제도 (SOCIAL_SECURITY)
  - 세금/금융 (TAX_FINANCE)
  - 의료/건강 (MEDICAL_HEALTH)
  - 취업 (EMPLOYMENT)
  - 일상생활 (DAILY_LIFE)
- RAG 기반 정확한 정보 제공
- 추론 기반 복잡한 질문 처리

## 개발 환경 설정

### 1. Python 환경 설정

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. Ollama 설정

Ollama는 LLM 모델을 로컬에서 실행하기 위한 도구입니다.

```bash
# Ollama 설치 (Mac)
brew install ollama

# Ollama 서버 실행
ollama serve

# 필요한 모델 다운로드
ollama pull gemma3:1b  # 경량 모델
ollama pull gemma3:12b  # 고성능 모델
```

### 3. 서버 실행

```bash
# 환경 변수 설정
cp .env.example .env
# .env 파일을 수정하여 필요한 설정을 입력

# 서버 실행
PYTHONPATH=$PYTHONPATH:. uvicorn app.main:app --reload
```

## API 엔드포인트

### 챗봇 API

```
POST /api/v1/chatbot
Content-Type: application/json

{
    "query": "건강보험 자격 취득은 어떻게 하나요?",
    "uid": "user_id"
}
```

### 에이전트 API

```
POST /api/v1/agentic
Content-Type: application/json

{
    "query": "건강보험 자격 취득은 어떻게 하나요?",
    "uid": "user_id"
}
```

## 개발 가이드

### 1. RAG 도메인 추가

1. `app/config/rag_config.py`에 새로운 도메인 추가
2. `app/services/rag_service.py`에 도메인별 문서 추가
3. `app/models/chatbot_response.py`에 RAG 유형 추가

### 2. 새로운 기능 추가

1. `app/services/` 디렉토리에 새로운 서비스 클래스 추가
2. `app/api/v1/` 디렉토리에 API 엔드포인트 추가
3. `app/models/` 디렉토리에 필요한 데이터 모델 추가

## 테스트

```bash
# 테스트 실행
pytest tests/
```

## 로깅

- 로그 파일: `server.log`
- 로그 레벨: INFO
- 주요 로그 포인트:
  - 전처리: 번역, 언어 감지
  - 분류: 질의 유형, RAG 유형
  - 응답 생성: RAG 컨텍스트, LLM 응답
  - 후처리: 번역, 응답 정제

## 서버 실행 방법
프로젝트 루트에서 다음 실행(백그라운드 실행)
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 &

서버 종료는 다음으로 실행
ps aux | grep uvicorn      # 실행 중인 프로세스 확인
kill -9 [PID]              # 프로세스 종료


서버 실행 중 로그는 프로젝트 루트의 uvicorn.log 확인
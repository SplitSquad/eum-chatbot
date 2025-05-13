# EUM Chatbot

한국에서 생활하는 외국인을 위한 챗봇 서비스입니다. 다양한 도메인의 정보를 제공하고, 다국어 지원을 통해 외국인들의 한국 생활을 돕습니다.

## 주요 기능

- **다국어 지원**: 한국어, 영어, 일본어, 중국어 등 다양한 언어 지원
- **도메인별 정보 제공**:
  - 비자/법률 (Visa/Law)
  - 사회보장제도 (Social Security)
  - 세금/금융 (Tax/Finance)
  - 의료/건강 (Medical/Health)
  - 취업 (Employment)
  - 일상생활 (Daily Life)
- **RAG 기반 응답**: 도메인별 벡터 데이터베이스를 활용한 정확한 정보 제공
- **실시간 웹 검색**: 최신 정보 제공을 위한 웹 검색 기능

## 기술 스택

- **Backend**: FastAPI
- **LLM**: Groq API (Gemma 3 12B)
- **Vector DB**: ChromaDB
- **Embedding**: Sentence Transformers (paraphrase-multilingual-mpnet-base-v2)
- **Language Processing**: LangChain

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd eum-chatbot
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 열어 필요한 API 키와 설정을 입력
```

## 실행 방법

1. 서버 실행
```bash
PYTHONPATH=$PYTHONPATH:. uvicorn app.main:app --reload
```

2. API 엔드포인트
- 챗봇 API: `POST /api/v1/chatbot`
  ```json
  {
    "query": "한국에서 필요한 기본 서류는?",
    "uid": "user_id"
  }
  ```

## 프로젝트 구조

```
eum-chatbot/
├── app/
│   ├── api/            # API 엔드포인트
│   ├── config/         # 설정 파일
│   ├── core/           # 핵심 기능
│   ├── models/         # 데이터 모델
│   ├── services/       # 비즈니스 로직
│   │   ├── chatbot/    # 챗봇 관련 서비스
│   │   └── common/     # 공통 서비스
│   └── main.py         # 애플리케이션 진입점
├── data/
│   ├── chroma/         # ChromaDB 데이터
│   └── vectorstore/    # 벡터 스토어 데이터
├── logs/               # 로그 파일
├── tests/              # 테스트 코드
├── .env.example        # 환경 변수 예시
├── requirements.txt    # 의존성 목록
└── README.md          # 프로젝트 문서
```

## 주요 서비스

1. **ChatbotService**: 챗봇의 메인 서비스
   - 질의 분류
   - 응답 생성
   - 다국어 처리

2. **RAGService**: 도메인별 정보 검색
   - 벡터 데이터베이스 관리
   - 관련 문서 검색
   - 컨텍스트 생성

3. **WebSearchService**: 실시간 웹 검색
   - 최신 정보 검색
   - 검색 결과 처리

## 응답 생성 프로세스

1. **질의 분류**
   - 질의 유형 분류 (일반, 추론, 웹 검색)
   - RAG 도메인 분류

2. **컨텍스트 생성**
   - RAG 기반 관련 문서 검색
   - 웹 검색 결과 수집 (필요시)

3. **응답 생성**
   - Groq API를 사용한 응답 생성
   - 다국어 번역 및 후처리

## 로깅

- 로그 파일 위치: `logs/app.log`
- 로그 레벨: INFO, ERROR
- 주요 로그 카테고리:
  - [WORKFLOW]: 전체 워크플로우 로그
  - [RESPONSE]: 응답 생성 관련 로그
  - [RAG]: RAG 관련 로그
  - [POSTPROCESS]: 후처리 관련 로그
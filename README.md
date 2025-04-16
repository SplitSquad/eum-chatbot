# eum-chatbot
대화형 유저 맞춤, 에이전틱 기능 수행 가능한 챗봇


## 서버 실행 방법
프로젝트 루트에서 다음 실행(백그라운드 실행)
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > uvicorn.log 2>&1 &

서버 종료는 다음으로 실행
ps aux | grep uvicorn      # 실행 중인 프로세스 확인
kill -9 [PID]              # 프로세스 종료


서버 실행 중 로그는 프로젝트 루트의 uvicorn.log 확인
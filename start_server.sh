#!/bin/bash

# 로그 디렉토리 확인 및 생성
if [ ! -d "logs" ]; then
  mkdir -p logs
fi

echo "Starting server on port 8000..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info --log-config app/config/log_config.json 
#!/usr/bin/env bash
# run.sh 로 띄운 데모 서버를 내린다. 프로세스 그룹째 종료하므로 그 아래
# ocr.py --serve 와 llama-server(GPU 점유)까지 같이 정리된다.
set -uo pipefail
cd "$(dirname "$0")"

PID_FILE=${PID_FILE:-.server.pid}

if [ ! -f "$PID_FILE" ]; then
  echo "실행 중이 아닙니다 ($PID_FILE 없음)."
else
  pid=$(cat "$PID_FILE")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "pid $pid 는 이미 죽어 있습니다. $PID_FILE 만 지웁니다."
    rm -f "$PID_FILE"
  else
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    echo "종료 중 (pid $pid, 그룹 $pgid)…"
    kill -TERM -"$pgid" 2>/dev/null || kill -TERM "$pid"
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "  응답이 없어 강제 종료합니다."
      kill -KILL -"$pgid" 2>/dev/null || kill -KILL "$pid"
    fi
    rm -f "$PID_FILE"
    echo "종료됨."
  fi
fi

# 서버가 SIGKILL 당했거나 비정상 종료한 뒤 남은 OCR llama-server 청소.
# 이 모델 경로를 쓰는, 내 소유의 프로세스만 건드린다(다른 llama-server는 그대로 둔다).
leftovers=$(pgrep -u "$(id -u)" -f "llama-server.*unlimited-ocr" || true)
if [ -n "$leftovers" ]; then
  echo "남아 있던 OCR llama-server 정리: $(echo "$leftovers" | tr '\n' ' ')"
  # shellcheck disable=SC2086
  kill $leftovers 2>/dev/null
  sleep 2
  # shellcheck disable=SC2086
  kill -KILL $leftovers 2>/dev/null
fi

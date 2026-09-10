#!/usr/bin/env bash
# 데모 서버를 백그라운드로 띄운다. setsid 로 터미널과 끊어 두므로 SSH 세션이
# 끝나도 계속 돈다(재부팅 후에는 다시 실행해야 한다).
#
# 기본값은 이 머신의 포트포워딩(NAT)에 맞춰져 있다: 내부 8081/5173 포트가
# 공인 IP 쪽 35800/35915 로 매핑되어 있다(service.gsmsv.site). PUBLIC_PORT는
# 그 매핑에서 "밖에 보이는" API 포트로, 브라우저가 API를 부를 주소를 만드는 데
# 쓰인다(config.js). PORT(바인딩 포트)와 달라도 되고, 오히려 NAT 환경에서는
# 달라야 한다 - 자세한 이유는 server.py --public-port 설명 참고.
#
#   ./run.sh                    # API 내부 8081(공인 35800), 화면 5173(공인 35915), GPU 0,1
#   GPUS=0,1,3 ./run.sh         # GPU 3장 = 동시 3건
#   PORT=9000 CLIENT_PORT=5174 ./run.sh
#   PUBLIC_PORT=9000 ./run.sh   # 포트포워딩 없이 그냥 로컬/직결이면 PORT와 맞춰서 지정
#   ./run.sh --no-ocr           # 그 밖의 인자는 server.py 로 그대로 넘어간다
set -euo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8081}
CLIENT_PORT=${CLIENT_PORT:-5173}
PUBLIC_PORT=${PUBLIC_PORT:-35800}
GPUS=${GPUS:-0,1}
LOG=${LOG:-server.log}
PID_FILE=${PID_FILE:-.server.pid}

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "이미 실행 중입니다 (pid $(cat "$PID_FILE")). 먼저 ./stop.sh 를 실행하세요." >&2
  exit 1
fi
rm -f "$PID_FILE"

for port in "$PORT" "$CLIENT_PORT"; do
  if ss -lnt "( sport = :$port )" | grep -q LISTEN; then
    echo "error: 포트 $port 를 이미 다른 프로세스가 쓰고 있습니다." >&2
    ss -lntp "( sport = :$port )" | tail -n +2 >&2
    exit 1
  fi
done

# setsid: 세션을 분리해 터미널이 닫혀도 살아남게 한다. 프로세스 그룹 리더가 되므로
# stop.sh 가 그룹째 종료해 자식(ocr.py --serve, llama-server)까지 같이 정리한다.
setsid nohup python3 server.py \
  --port "$PORT" --client-port "$CLIENT_PORT" --public-port "$PUBLIC_PORT" --gpus "$GPUS" \
  --pid-file "$PID_FILE" "$@" >>"$LOG" 2>&1 </dev/null &

for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
    host=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "시작됨 (pid $(cat "$PID_FILE" 2>/dev/null || echo '?'))"
    echo "  화면 : http://localhost:$CLIENT_PORT${host:+  (http://$host:$CLIENT_PORT)}"
    echo "  공인 : http://service.gsmsv.site:35915  (API는 :$PUBLIC_PORT 로 나감)"
    echo "  API  : http://localhost:$PORT/api/status"
    echo "  로그 : $LOG        중지: ./stop.sh"
    exit 0
  fi
  sleep 1
done

echo "error: 서버가 30초 안에 뜨지 않았습니다. $LOG 를 확인하세요." >&2
tail -n 20 "$LOG" >&2 || true
exit 1

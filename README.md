# OCR-demo

한국 중학교 학교생활세부사항기록부 PDF를 OCR로 읽고, 그 표를 미리 정해둔 서식에
맞춰 JSON으로 뽑아내는 파이프라인.

- `ocr.py` : PDF -> OCR 텍스트 (CUDA + llama.cpp, Unlimited-OCR GGUF)
- `extract.py` : OCR 텍스트 -> 구조화 JSON (규칙 기반 표 파서, LLM 없음)
- `reconstruct.py` : OCR 좌표로 원본 레이아웃을 재현한 HTML/PDF (검수용)
- `pipeline.py` : 위 세 스크립트를 순서대로 실행하는 래퍼
- `server.py` + `web/` : PDF를 올리면 파싱 결과를 보여주는 데모 웹사이트

## 요구 사항

- CUDA가 있는 리눅스 머신, `~/llama.cpp`에 `-DGGML_CUDA=ON`으로 빌드된
  `build/bin/llama-server`
- 시스템 `python3`에 `requirements.txt` 의존성 설치 (가상환경 없음)
- Unlimited-OCR GGUF 모델 (최초 1회):
  ```bash
  hf download sahilchachra/Unlimited-OCR-GGUF \
    Unlimited-OCR-BF16.gguf mmproj-Unlimited-OCR-F16.gguf \
    --local-dir ~/llama.cpp/models/unlimited-ocr
  ```

추출 단계는 LLM도 인터넷도 쓰지 않는다. 표만 읽으면 되므로 17페이지 문서가
0.5초 안에 끝난다.

## 추출 방식

`extract.py`는 이 서식에 나오는 표를 `TEMPLATES`에 **표 서식**으로 선언해 두고,
표가 나올 때마다 머리글을 대조해 어느 서식인지 판정한 뒤 그 서식이 정한 칸만
읽는다. OCR이 표를 흔들어 놓는 세 가지 경우를 모두 무르게 처리한다.

1. **머리글 대조**는 글자가 정확히 같지 않아도 된다 (`답임성명`/`담임성명`,
   `수업입안수`/`수업일수`).
2. **칸 정렬**은 rowspan/colspan을 편 격자 기준과 셀 순서 기준 두 가지 후보를
   만들어, 칸마다 정해둔 값 검증기(학년·날짜·원점수·성취도…)로 점수를 매겨
   이긴 쪽을 쓴다. 빈 열이 끼어들거나 앞 페이지에서 이어지느라 첫 칸이 빠져도
   견딘다.
3. **머리글이 깨진 표**(3학년 2학기 성적표처럼 한자로 뭉개진 경우)는 같은 절에서
   마지막으로 읽은 서식이 이어지는 것으로 보고, 값 검증을 통과한 행만 살린다.

서술형(특기사항, 세부능력 및 특기사항, 행동특성 및 종합의견)은 뽑지 않는다.
어떤 표를 어느 서식으로 읽었는지는 `--verbose`로 확인할 수 있다.

## 데모 웹사이트

```bash
./run.sh              # 백그라운드 실행 (화면 :5173, API :8081, GPU 0,1)
./stop.sh             # 종료 (llama-server까지 같이 정리)
tail -f server.log    # 로그
```

`run.sh`는 `setsid`로 터미널과 끊어서 띄우므로 **SSH 세션을 닫아도 계속 떠 있다**
(재부팅하면 다시 실행해야 한다). 환경변수로 바꿀 수 있다:

```bash
GPUS=0,1,3 ./run.sh                    # GPU 3장 = PDF 3개 동시 처리
PORT=9000 CLIENT_PORT=5174 ./run.sh    # 포트 변경
./run.sh --no-ocr                      # 나머지 인자는 server.py 로 그대로 전달
python3 server.py                      # 포그라운드로 직접 실행 (같은 기본 포트)
```

브라우저에서 <http://localhost:5173> 을 열고 PDF를 여러 개 끌어다 놓으면 페이지
단위 진행률이 보이고, 끝나면 인적사항·출결·교과 성적·수상·창체·봉사·자유학기·
독서가 표로 뜬다. JSON은 그 자리에서 내려받을 수 있고 OCR 원문도 열어볼 수 있다.
문서당 걸린 시간은 모델 로딩/OCR/추출로 나뉘어 표시된다.

- 포트는 둘이지만 프로세스는 하나다. `--port 8081`은 JSON API,
  `--client-port 5173`은 `web/`을 그대로 내보내는 정적 서버다. 화면이 API를
  찾는 방법은 서버가 실제 포트로 만들어 주는 `config.js`(`window.API_PORT`)뿐이라
  플래그만 바꾸면 따라온다. `?api=http://호스트:포트`로 덮어쓸 수도 있다.
- API는 CORS를 전부 열어 둔다(`allow_origins=["*"]`). 인증이 없는 데모라 오리진을
  좁혀도 얻는 게 없지만, 그만큼 신뢰할 수 없는 망에 그대로 열지 말 것.

- 업로드는 **uuid4 작업**이 되어 `runs/<uuid>/`(`--work-dir`)에
  `input.pdf`/`ocr.txt`/`extracted.json`으로 따로 담긴다. 파일명이 겹치지 않으므로
  같은 문서를 동시에 여러 개 올려도 서로를 덮어쓰지 않는다. 올린 파일 이름은
  화면에 보여줄 이름표로만 쓴다.
- 동시 처리 수는 디스크가 아니라 **GPU 수**가 정한다. `--gpus` 번호 하나당 워커
  하나가 큐에서 작업을 꺼내고, 워커마다 자기 `ocr.py --serve` llama-server를
  띄워 두므로 5.9GB 모델은 GPU당 한 번만 올라간다(문서마다 다시 올리지 않는다).
  워커 수를 넘는 업로드는 VRAM을 두고 다투는 대신 큐에서 기다린다.
- `server.py`는 파이프라인을 다시 구현하지 않는다. CLI와 똑같은 `ocr.py`/
  `extract.py`를 서브프로세스로 부르므로 웹 결과와 CLI 결과가 갈라질 수 없다.
- API: `POST /api/jobs`(multipart, 여러 개 가능) / `GET /api/jobs` /
  `GET /api/jobs/{id}` / `GET /api/jobs/{id}/json` / `GET /api/jobs/{id}/ocr` /
  `DELETE /api/jobs/{id}` / `GET /api/status`. 문서는 `:8081/api/docs`.
- 데모용이라 인증이 없고 작업 목록은 메모리에만 있다(서버를 내리면 사라진다).
- 17페이지 문서 1건 기준 실측: 모델 로딩 17초(그 GPU의 첫 작업만) + OCR 241초
  (14.2초/쪽) + 추출 0.4초. GPU가 다르면 서로 느려지지 않는다(2건 동시 246·248초).

## 실행

```bash
# 전체 파이프라인 (OCR -> 추출)
python3 pipeline.py document.pdf

# 이미 만들어둔 document_ocr.txt 가 있으면 OCR 생략하고 추출만 (개발 중 기본)
python3 pipeline.py --skip-ocr

# 추출 후 레이아웃 재현 HTML/PDF도 같이 생성
python3 pipeline.py document.pdf --reconstruct

# OCR 단독 실행 (stdout이 OCR 텍스트이므로 반드시 리다이렉트)
python3 ocr.py document.pdf > document_ocr.txt

# 이미 떠 있는 llama-server를 재사용 (새로 안 띄움)
python3 ocr.py document.pdf --server-url http://<host>:<port> > document_ocr.txt

# 추출 단독 실행
python3 extract.py --input document_ocr.txt --output document_extracted.json

# 어떤 표를 어느 서식으로 읽었는지 표별로 확인 (서식 점검용)
python3 extract.py --verbose

# 레이아웃 재현 (Chrome 없으면 PDF 생략)
python3 reconstruct.py document.pdf --ocr-text document_ocr.txt --no-pdf
```

- `ocr.py`는 결과를 파일에 쓰지 않는다. OCR 텍스트는 stdout, 진행 로그는
  stderr로 나간다. `pipeline.py`가 이 리다이렉트와 임시파일 처리를 대신한다.
- 17페이지 기준 OCR에 수 분 걸리므로, `document_ocr.txt`가 이미 있으면
  재실행하지 말고 `--skip-ocr`로 재사용할 것.
- `ocr.py`는 한자가 든 토큰을 샘플러에서 기본으로 막는다(`--ban-han`, 기본 켬). 이 모델은
  중국어 토큰 쪽으로 한 번 미끄러지면 페이지 전체를 한자로 채워버리는데, 막으면 같은
  이미지에서 정확한 한국어가 나온다(이 문서 기준 한자 3,958자 -> 0자, 속도 동일).
  성명을 한자로 병기하는 서식이라면 `--no-ban-han`.
- `ocr.py`의 `--header-frac`/`--footer-frac` 기본값은 특정 서식의 도장·QR·
  바코드 위치에 맞춘 값이라 다른 서식에는 다시 조정해야 한다.
- `reconstruct.py`는 Chrome 실행 경로를 하드코딩해 PDF를 뽑는다. Chrome이
  없으면 `--no-pdf`로 HTML만 만들 것.
- `config.py`/`.env`/`.env.example`은 LLM으로 추출하던 시절의 설정이라 지금은
  어느 스크립트도 읽지 않는다.

자세한 내부 동작·표 서식 수정 규칙·OCR 텍스트 형식은 `CLAUDE.md` 참고.

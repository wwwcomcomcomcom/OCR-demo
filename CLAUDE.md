# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

한국 중학교 학교생활세부사항기록부 PDF를 OCR로 읽고(`ocr.py`), 그 표를 미리 선언해 둔
표 서식에 맞춰 JSON으로 뽑는(`extract.py`) 파이프라인. 추출 단계는 LLM을 쓰지 않는
규칙 기반 파서다(17페이지 0.2초). `reconstruct.py`는 OCR 좌표로 원본 레이아웃을
재현한 HTML/PDF를 만드는 검수용 도구. `server.py` + `web/`는 PDF를 업로드하면 이
파이프라인을 돌려 결과를 보여주는 데모 웹사이트다.

## 실행

```bash
python3 pipeline.py document.pdf        # OCR -> 추출 전체
python3 pipeline.py --skip-ocr          # 기존 document_ocr.txt 재사용 (개발 중 기본)
python3 ocr.py document.pdf > document_ocr.txt   # OCR 단독 실행 시 리다이렉트 필수
python3 extract.py --verbose            # 어떤 표를 어느 서식으로 읽었는지 표별로 출력
./run.sh                                # 데모 웹서버 백그라운드 실행 (5173/8081)
./stop.sh                               # 그 서버와 llama-server까지 종료
```

- `ocr.py`는 결과를 파일에 쓰지 않는다. **OCR 텍스트는 stdout, 진행 로그는 stderr**로
  나가므로 단독 실행 시 반드시 stdout을 리다이렉트한다. `pipeline.py`가 이 리다이렉트와
  임시파일 처리를 대신한다.
- OCR은 CUDA + `~/llama.cpp`(빌드된 `llama-server`)로 Unlimited-OCR GGUF(BF16)를 돌린다.
  `ocr.py`가 이 서버를 서브프로세스로 직접 띄우고 끝나면 종료하므로 별도로 서버를 미리
  실행해 둘 필요는 없다(이미 떠 있는 서버를 쓰려면 `--server-url` 지정). GPU는
  `CUDA_VISIBLE_DEVICES`로 0번에 고정되며 `--gpu-id`로 바꿀 수 있다. 17페이지 기준 수 분
  걸리므로 `document_ocr.txt`가 이미 있으면 재실행하지 말고 `--skip-ocr`로 재사용할 것.
- OCR 샘플러는 **한자가 든 토큰을 기본으로 금지한다**(`ocr.py`의 `--ban-han`, 끄려면
  `--no-ban-han`). 이 모델은 한 번 중국어 토큰 쪽으로 미끄러지면 되돌아오지 못하고
  그 페이지를 한자로 채워버린다(금지 전 12페이지: `학기|교과|과목` -> `釣り|豆群|群号`,
  같은 글자를 토큰 상한까지 반복). 금지하면 같은 이미지에서 정확한 한국어가 나오고
  속도 손해는 없다(문서 전체 한자 3,958자 -> 0자). 금지 목록은 GGUF의
  `tokenizer.ggml.tokens`를 직접 읽어 만든다(vocab 129,280개 중 35,353개, 0.5초).
  DeepSeek-v3 BPE는 byte-level 매핑이라 토큰 문자열을 그냥 UTF-8로 풀면 한자가 하나도
  안 잡힌다 - `byte_decoder()`로 되돌려야 한다. 성명을 한자로 병기하는 서식에는 끌 것.
- 모델 파일은 저장소에 없다. 최초 1회만 받으면 된다:
  ```bash
  hf download sahilchachra/Unlimited-OCR-GGUF \
    Unlimited-OCR-BF16.gguf mmproj-Unlimited-OCR-F16.gguf \
    --local-dir ~/llama.cpp/models/unlimited-ocr
  ```
  `--model-path`/`--mmproj-path`로 다른 경로를 쓸 수 있다.

## 환경

- 가상환경이 없다. 의존성은 `requirements.txt`에 있고 시스템
  `python3`(`/usr/bin/python3`)에 직접 설치되어 있다. `python3`로 실행할 것.
- `ocr.py`는 CUDA가 있는 리눅스 머신을 전제로 `~/llama.cpp`의 빌드 결과물
  (`build/bin/llama-server`)을 서브프로세스로 사용한다. llama.cpp는 이 저장소에 포함되어
  있지 않으므로 CUDA(`-DGGML_CUDA=ON`)로 따로 빌드해 둬야 한다.
- 입력 문서와 산출물(`*.pdf`, `*.png`, `*_ocr.txt`, `*_extracted.json`, 재현 HTML,
  `*.log`)은 `.gitignore`로 제외되어 있다. 커밋 대상은 코드와 설정뿐이다.

## 데모 웹서버 (`server.py` + `web/`, `run.sh`/`stop.sh`)

- 포트 둘, 프로세스 하나다. `--port 8081`은 FastAPI(JSON API), `--client-port 5173`은
  `web/`을 내보내는 표준 라이브러리 정적 서버(스레드)다. 화면이 API 포트를 아는 방법은
  서버가 만들어 주는 `config.js`(`window.API_PORT`) 하나뿐이니, 포트를 옮길 때 JS에
  주소를 박아 넣지 말 것. 호스트는 페이지를 연 주소를 그대로 쓴다(IP·터널에서도 동작).
- 이 전제는 브라우저가 접속한 포트와 서버가 바인딩한 포트가 같을 때만 성립한다.
  포트포워딩/NAT로 외부 포트가 내부와 다르면(`run.sh` 기본값: 내부 8080/5173 ->
  공인 34762/35915) `config.js`가 내부 포트를 그대로 알려줘 버려서 브라우저가 열리지
  않는 포트로 API를 부르게 된다. `--public-port`(= `run.sh`의 `PUBLIC_PORT`)로 밖에서
  보이는 API 포트를 따로 알려줘야 한다.
- API 포트로도 같은 화면이 열린다(StaticFiles를 `/`에 마지막으로 mount). 포트 하나만
  열 수 있는 환경을 위한 것이므로 `/api/...` 경로와 겹치는 라우트를 뒤에 추가하지 말 것.
- `run.sh`는 `setsid`로 세션을 분리해 띄우고 `--pid-file`에 PID를 남긴다. `stop.sh`는
  그 PID의 **프로세스 그룹째** 종료해 `ocr.py --serve`와 llama-server까지 정리한다.
  그래서 `run.sh`에서 `setsid`를 빼면 stop.sh가 GPU를 반납하지 못한다.
- `ocr.py --serve`는 SIGTERM을 받으면 `sys.exit`로 빠져나가 llama-server를 정리한다.
  이 핸들러가 없으면 서버를 내려도 llama-server가 GPU를 11.5GB씩 물고 남는다.
- 파이프라인을 다시 구현하지 않는다. `pipeline.py`와 마찬가지로 작업마다 `ocr.py`와
  `extract.py`를 **서브프로세스로** 부른다. 파싱 규칙을 고칠 일이 생기면 `extract.py`만
  고치면 웹에도 그대로 반영된다. 서버 쪽에 표 해석 로직을 두지 말 것.
- 업로드 1건 = uuid4 작업 1개 = `--work-dir`(기본 `runs/`) 아래 디렉터리 1개
  (`input.pdf`/`ocr.txt`/`extracted.json`). `document_ocr.txt` 같은 공용 파일명을 쓰지
  않으므로 여러 문서를 동시에 처리해도 서로 덮어쓰지 않는다. 업로드된 파일명은 저장
  경로에 절대 쓰지 않는다(표시용 이름표일 뿐 — 경로 조작 방지).
- 동시성의 한계는 디스크가 아니라 GPU다. `--gpus` 번호 하나당 워커 스레드 하나가 공용
  큐에서 작업을 꺼내며, 워커마다 자기 llama-server(`ocr.py --serve`, 첫 작업 때 기동)를
  들고 있어 5.9GB 모델을 GPU당 한 번만 올린다. 워커를 GPU 수보다 늘리면 VRAM이 터진다.
- `ocr.py --serve`는 이 서버를 위해 있는 모드다. llama-server만 띄우고 대기하며, 실제
  OCR 실행은 `--server-url`로 그 서버에 붙는다. `--serve`를 지우면 문서마다 모델을
  다시 로딩하게 된다.
- 진행률은 `ocr.py`가 stderr에 찍는 `[N page(s)]` / `[OCR page i/N]` 줄을 정규식으로
  읽어 만든다(`RE_PAGES`/`RE_PAGE`). 그 로그 문구를 바꾸면 진행 표시가 멈춘다.
- 프런트엔드는 의존성 없는 순수 JS(`web/app.js`)이고 1.2초마다 `/api/jobs`를 폴링한다.
  OCR 텍스트를 화면에 넣을 때는 `innerHTML` 대신 `textContent`(`h()` 헬퍼)를 쓴다.
- 데모용이라 인증이 없고 작업 목록은 프로세스 메모리에만 있다. 재시작하면 목록은
  사라지지만 `runs/`의 파일은 남는다(`--reset`으로 비운다).
- GPU 없이 UI만 손볼 때는 `--no-ocr`. OCR 단계를 건너뛰고 `--ocr-text`(기본
  `document_ocr.txt`)를 그 작업의 OCR 결과로 복사해 추출만 돌린다.

## 남아 있는 LLM 설정 파일

`config.py` / `.env` / `.env.example`은 추출을 LLM으로 하던 시절의 잔재다. 지금은 어느
스크립트도 읽지 않는다(그래서 `openai` 의존성도 requirements에서 뺐다). 새 설정을 여기에
추가하지 말 것.

## `extract.py` 표 서식 수정 규칙

`TEMPLATES`에 표 하나당 `Template` 하나를 선언한다. 표가 나올 때마다 모든 서식의
머리글과 대조해 가장 잘 맞는 서식으로 읽고, 그 서식이 정한 칸만 뽑는다.

- `labels`는 **머리글 서식**(이 표인지 판정하는 기준), `columns`는 **뽑을 칸**이다.
  둘은 다를 수 있다: 출결표는 머리글에 `결석일수`/`질병`이 따로 있지만 칸은
  `결석일수 질병`처럼 합쳐서 잡는다. `labels`를 비워두면 컬럼 라벨을 그대로 쓴다.
- 대조는 정확 일치가 아니라 유사도(`MIN_LABEL_SIMILARITY`)다. OCR이 `담임성명`을
  `답임성명`으로, `수업일수`를 `수업입안수`로 읽어도 붙는다. 임계값을 올리면 이런
  표를 놓치고, 내리면 비슷한 표끼리 서로 먹는다(공통교과 vs 체육·예술).
- 각 `Column`의 `kind`는 값 검증기이자 **정렬 판정 기준**이다. 한 행을 격자 기준과
  셀 순서 기준 두 가지로 읽어 보고 `kind`를 더 많이 만족하는 쪽을 택하므로, 새 칸을
  넣을 때 `text` 말고 맞는 `kind`를 주는 것이 곧 정확도다. 검증에 걸린 값은 버린다.
- `carry=True`(학년·학기)는 앞 행 값을 물려받는다는 뜻이고, rowspan으로 넘어온 셀이
  있으면 그쪽이 우선이다. `key=True` 칸이 모두 비면 그 행은 항목으로 만들지 않는다.
- 표를 새로 추가하면 `Extractor.emit()`에 그 서식의 분기와 `summarize()` 집계도 같이
  넣어야 한다. `emit()`은 `Template.name` 앞부분(절 번호)으로 분기한다.
- 이 서식들은 정량화된 '값'만 담고 서술형(특기사항, 세부능력 및 특기사항, 행동특성 및
  종합의견)은 담지 않는다. `is_junk()`가 300자 넘는 칸을 버리므로, 서술형을 담으려면
  그 컷부터 풀어야 한다.
- 고친 뒤에는 반드시 `--verbose`로 표별 판정 로그를 보고, 놓친 표(`서식 미상`)와
  잘못 붙은 표가 없는지 확인할 것. 이 문서 기준 정상값은 학적 3 / 출결 3 / 교과 64 /
  수상 2 / 창의적 9 / 봉사 28 / 자유학기 8 / 독서 2 건이고, 서식에 없는 표 13개
  (세부능력 및 특기사항, 행동특성, 발급 정보)는 건너뛴다.

## OCR 텍스트 형식 (`document_ocr.txt`)

- 페이지 구분자 `===== Page N =====`, 표는 `<table>` HTML, 블록마다
  `<|det|>type [x0, y0, x1, y1]<|/det|>` 좌표 태그가 붙는다.
- CUDA/llama.cpp로 바꾼 뒤 합성 테스트 이미지로는 `<|det|>...<|/det|>` 태그가 그대로
  나오는 것을 확인했다(`llama-server`에 `--special` 필요 — 없으면 이 태그들이 조용히
  사라진다). 다만 실제 학교생활기록부 PDF로는 재확인 못 했으니(테스트 환경에 샘플
  PDF가 없었다), 처음 실행한 `document_ocr.txt`는 태그가 여전히 붙어 나오는지 한 번
  눈으로 확인할 것. 안 나오면 `reconstruct.py`의 레이아웃 재현이 어긋난다.
- `reconstruct.py`가 이 좌표 태그를 파싱해 레이아웃을 재현한다. 형식을 바꾸면 깨진다.
- `extract.py`는 이 태그를 블록 구분에만 쓰고 좌표는 보지 않는다. 태그가 통째로
  빠진 입력(`--special` 없이 돌린 OCR)에서도 `<table>` 단위로 쪼개 같은 결과를 낸다.
- OCR이 같은 표나 행을 되풀이하며 페이지를 채우는 일이 있다(이 문서에서는 7·9·12페이지).
  `temperature 0`에서는 같은 문맥이 같은 토큰을 내므로 한 번 빠지면 못 나오고, 막지 않으면
  그 페이지가 토큰 상한(`--max-tokens 8192`)을 다 쓴다 — 이 문서 전체 디코딩의 3분의 2가
  이 세 페이지였다. 그래서 `ocr.py`는 생성을 스트리밍으로 받아 꼬리가 같은 블록의 반복이
  되면(`repeating_tail`) 연결을 끊어 생성을 중단하고, 그 페이지만 `temperature 0.3`으로
  다시 읽는다. 임계값(`LOOP_MIN_SPAN` 등)을 낮추면 빈 칸이 반복되는 정상 서식 표를
  잘라먹으니, 고치면 정상 페이지에서 오탐이 없는지 반드시 확인할 것.
- 그래도 반복 꼬리는 남을 수 있다. `extract.py`는 같은 표가 연달아 나오면 한 번만
  읽고, 같은 글자가 10자 이상 반복된 칸은 버리며, 교과 성적은 학년/학기/영역/과목이
  같으면 뒤엣것을 버린다(반복 꼬리에서 교과명만 뭉개진 근사-중복 행이 나온다).
- rowspan/colspan 때문에 표 셀이 밀려 있거나 `</table>`이 빠진 페이지가 있고, 여백의
  눈금 숫자가 텍스트로 섞여 들어온다. 정상이며 프롬프트에서 처리한다.

## 문서 레이아웃 의존 값

- `ocr.py`의 `--header-frac 0.115` / `--footer-frac 0.82` 기본값은 이 문서의 도장·QR·
  바코드 위치에 맞춘 값이다. 다른 서식에는 다시 조정해야 한다.
- `reconstruct.py`는 Chrome 실행 경로(`/Applications/Google Chrome.app/...`)를
  하드코딩해 PDF를 뽑는다. Chrome이 없으면 `--no-pdf`로 HTML만 만들 것.

## 코드 스타일

- 모듈 docstring과 주석은 영어, CLI `help`·LLM 프롬프트·사용자에게 보이는 출력은 한국어.
- 각 스크립트는 argparse CLI를 갖는 단일 파일이고 서로 import하지 않는다
  (`pipeline.py`도 서브프로세스로 호출만 한다). 공유 설정만 `config.py`에 둔다.
- 진행 상황·경고는 stderr, 파이프로 넘길 결과물은 stdout.

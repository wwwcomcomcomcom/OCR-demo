# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

한국 중학교 학교생활세부사항기록부 PDF를 OCR로 읽고(`ocr.py`), 그 표를 미리 선언해 둔
표 서식에 맞춰 JSON으로 뽑는(`extract.py`) 파이프라인. 추출 단계는 LLM을 쓰지 않는
규칙 기반 파서다(17페이지 0.2초). `reconstruct.py`는 OCR 좌표로 원본 레이아웃을
재현한 HTML/PDF를 만드는 검수용 도구.

## 실행

```bash
python3 pipeline.py document.pdf        # OCR -> 추출 전체
python3 pipeline.py --skip-ocr          # 기존 document_ocr.txt 재사용 (개발 중 기본)
python3 ocr.py document.pdf > document_ocr.txt   # OCR 단독 실행 시 리다이렉트 필수
python3 extract.py --verbose            # 어떤 표를 어느 서식으로 읽었는지 표별로 출력
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
- OCR이 토큰 상한(`--max-tokens 8192`)까지 같은 표나 행을 되풀이하며 페이지를 채우는
  일이 있다(이 문서에서는 7·9·12페이지). `extract.py`는 같은 표가 연달아 나오면 한 번만
  읽고, 같은 글자가 10자 이상 반복된 칸은 버리며, 교과 성적은 학년/학기/영역/과목이
  같으면 뒤엣것을 버린다(반복 꼬리에서 교과명만 뭉개진 근사-중복 행이 나온다).
  다만 OCR 시간 자체는 이미 쓴 뒤이므로 페이지가 유난히 길면 원문을 한 번 확인해 볼 것.
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

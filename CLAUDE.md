# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

한국 중학교 학교생활세부사항기록부 PDF를 OCR로 읽고(`ocr.py`), 그 텍스트를 LLM으로
정해진 JSON 스키마에 매핑하는(`extract.py`) 3단계 파이프라인. `reconstruct.py`는 OCR
좌표로 원본 레이아웃을 재현한 HTML/PDF를 만드는 검수용 도구.

## 실행

```bash
python3 pipeline.py document.pdf        # OCR -> 추출 전체
python3 pipeline.py --skip-ocr          # 기존 document_ocr.txt 재사용 (개발 중 기본)
python3 ocr.py document.pdf > document_ocr.txt   # OCR 단독 실행 시 리다이렉트 필수
python3 extract.py --print-prompt        # API 호출 없이 프롬프트만 점검
```

- `ocr.py`는 결과를 파일에 쓰지 않는다. **OCR 텍스트는 stdout, 진행 로그는 stderr**로
  나가므로 단독 실행 시 반드시 stdout을 리다이렉트한다. `pipeline.py`가 이 리다이렉트와
  임시파일 처리를 대신한다.
- OCR은 Apple Silicon MLX 전용이고 17페이지에 수 분 걸린다. `document_ocr.txt`가 이미
  있으면 재실행하지 말고 `--skip-ocr`로 재사용할 것.

## 환경

- 가상환경이 없다. 의존성은 `requirements.txt`에 있고 시스템
  `python3`(`/usr/bin/python3`)에 직접 설치되어 있다. `python3`로 실행할 것.
- `mlx-vlm`은 Apple Silicon 전용이라 다른 플랫폼에서는 설치가 실패한다. OCR 없이
  추출 단계만 쓸 거면 그 줄을 빼고 설치하면 된다.
- 입력 문서와 산출물(`*.pdf`, `*.png`, `*_ocr.txt`, `*_extracted.json`, 재현 HTML,
  `*.log`)은 `.gitignore`로 제외되어 있다. 커밋 대상은 코드와 설정뿐이다.

## LLM 서버 설정

- 설정 키는 `config.py`에 모여 있다 (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` /
  `LLM_MAX_OUTPUT_TOKENS` / `LLM_RESPONSE_FORMAT`). 값은 **실제 환경변수 → 프로젝트
  루트 `.env` → `config.py` 기본값** 순으로 결정된다.
- `.env`는 커밋하지 않는다. 항목을 추가·변경하면 `.env.example`도 같이 고칠 것.
  `.env` 파서는 `config.py`에 직접 들어 있다(python-dotenv 의존성 없음). 새 설정을
  추가할 때는 `os.environ`을 직접 읽지 말고 `config.setting()`을 쓴다.
- 자체 호스팅 서버가 스키마 강제(guided decoding)를 못 하면
  `--response-format json_object`로 내리면 스키마를 프롬프트에 실어 보낸다.

## `extract.py` 스키마 수정 규칙

`SCHEMA`는 OpenAI Structured Outputs **strict** 규격을 지켜야 서버가 거부하지 않는다.
필드를 추가·변경할 때:

- 모든 object에 `"additionalProperties": False`, 모든 프로퍼티를 `required`에 나열
  (선택 필드라는 개념이 없다).
- 값이 없을 수 있는 필드는 `"type": ["string", "null"]`처럼 null 유니온으로 표현.
- 중첩은 5단계까지. 스키마를 고치면 `SYSTEM_PROMPT`의 한글 라벨 매핑 설명과
  `summarize()`의 집계 항목도 같이 손볼 것.

## OCR 텍스트 형식 (`document_ocr.txt`)

- 페이지 구분자 `===== Page N =====`, 표는 `<table>` HTML, 블록마다
  `<|det|>type [x0, y0, x1, y1]<|/det|>` 좌표 태그가 붙는다.
- `reconstruct.py`가 이 좌표 태그를 파싱해 레이아웃을 재현하고 `extract.py`
  프롬프트도 이 형식을 전제한다. 형식을 바꾸면 두 스크립트가 같이 깨진다.
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

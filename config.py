"""LLM 서버 접속 설정.

자체 호스팅 서버(vLLM / SGLang / llama.cpp / Ollama 등 OpenAI 호환 엔드포인트)를
쓰기 위한 설정을 한곳에 모아둔 파일.

값의 우선순위는 다음과 같다:

    1. 실제 환경변수        (export LLM_API_KEY=...)
    2. 이 파일 옆의 .env    (ENV_FILE 환경변수로 다른 경로 지정 가능)
    3. 아래의 기본값

.env 는 커밋하지 않는다(.gitignore 처리됨). 형식은 .env.example 참고.
"""

import os

ENV_PATH = os.environ.get("ENV_FILE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".env"
)


def _load_env_file(path: str) -> dict[str, str]:
    """.env 를 파싱한다. python-dotenv 없이 동작하도록 최소 기능만 구현.

    지원: KEY=VALUE, 앞의 'export ', # 주석, 빈 줄, 값 감싸는 따옴표.
    """
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return values
    except OSError as e:
        print(f"warning: {path} 를 읽을 수 없습니다: {e}")
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


_ENV = _load_env_file(ENV_PATH)


def setting(name: str, default: str = "") -> str:
    """환경변수 -> .env -> 기본값 순서로 설정값을 찾는다."""
    return os.environ.get(name) or _ENV.get(name) or default


# OpenAI 호환 API의 베이스 URL. 끝에 /v1 까지 붙인다.
#   vLLM/SGLang: http://<host>:8000/v1
#   llama.cpp  : http://<host>:8080/v1
#   Ollama     : http://<host>:11434/v1
#   OpenAI 본사 : https://api.openai.com/v1
LLM_BASE_URL = setting("LLM_BASE_URL") or setting(
    "OPENAI_BASE_URL", "http://localhost:8000/v1"
)

# API 키. 자체 서버가 인증을 안 하면 아무 문자열이나 넣으면 된다(빈 값은 불가).
LLM_API_KEY = setting("LLM_API_KEY") or setting("OPENAI_API_KEY", "not-needed")

# 서버에 로드된 모델 이름 (vLLM은 보통 HF 리포지토리 경로 그대로).
LLM_MODEL = setting("LLM_MODEL", "gpt-4.1")

# 응답 최대 토큰 수. 서술형 원문을 그대로 옮기므로 넉넉해야 한다.
LLM_MAX_OUTPUT_TOKENS = int(setting("LLM_MAX_OUTPUT_TOKENS", "32768"))

# 구조화 출력 방식.
#   json_schema : 스키마를 서버가 강제 (vLLM/SGLang의 guided decoding, OpenAI 등)
#   json_object : JSON 형식만 강제하고 스키마는 프롬프트로 전달 (구형 서버 호환용)
#   none        : 아무 제약 없이 프롬프트로만 요청 (최후의 수단)
LLM_RESPONSE_FORMAT = setting("LLM_RESPONSE_FORMAT", "json_schema")

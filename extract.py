"""Extract structured fields from an OCR'd 학교생활세부사항기록부 with the OpenAI API.

Takes the page-by-page OCR text produced by `ocr.py` (tables kept as HTML) and
asks a text LLM to map it onto a fixed JSON schema (학생정보 / 교과·비교과 성적 /
출결현황), using Structured Outputs so the response shape is guaranteed.

The OCR step ("보기") and the schema mapping step ("정리하기") are deliberately
separate: the vision model only has to read, this step only has to organise.
"""

import argparse
import json
import sys

from openai import OpenAI

import config


def _counts(label: str) -> dict:
    """Schema for a 질병/미인정/기타 triplet (결석·지각·조퇴·결과일수)."""
    return {
        "type": "object",
        "description": label,
        "properties": {
            "illness": {"type": ["integer", "null"], "description": "질병"},
            "unexcused": {"type": ["integer", "null"], "description": "미인정"},
            "other": {"type": ["integer", "null"], "description": "기타"},
        },
        "required": ["illness", "unexcused", "other"],
        "additionalProperties": False,
    }


SCHEMA = {
    "type": "object",
    "properties": {
        "student": {
            "type": "object",
            "description": "1. 인적·학적사항",
            "properties": {
                "name": {"type": ["string", "null"], "description": "성명"},
                "gender": {"type": ["string", "null"], "description": "성별"},
                "resident_registration_number": {
                    "type": ["string", "null"],
                    "description": "주민등록번호",
                },
                "address": {"type": ["string", "null"], "description": "주소"},
                "school_history": {
                    "type": "array",
                    "description": "학적사항의 입학·졸업 이력",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": ["string", "null"], "description": "연월일"},
                            "description": {
                                "type": "string",
                                "description": "예: OO중학교 제1학년 입학",
                            },
                        },
                        "required": ["date", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "name",
                "gender",
                "resident_registration_number",
                "address",
                "school_history",
            ],
            "additionalProperties": False,
        },
        "attendance": {
            "type": "array",
            "description": "2. 출결상황 (학년별 1행)",
            "items": {
                "type": "object",
                "properties": {
                    "grade": {"type": ["integer", "null"], "description": "학년"},
                    "school_days": {"type": ["integer", "null"], "description": "수업일수"},
                    "absence": _counts("결석일수"),
                    "late": _counts("지각"),
                    "early_leave": _counts("조퇴"),
                    "result": _counts("결과"),
                    "remarks": {"type": ["string", "null"], "description": "특기사항"},
                },
                "required": [
                    "grade",
                    "school_days",
                    "absence",
                    "late",
                    "early_leave",
                    "result",
                    "remarks",
                ],
                "additionalProperties": False,
            },
        },
        "academic_records": {
            "type": "array",
            "description": (
                "5. 교과학습발달상황의 성적 표 (공통교과, 〈체육·예술(음악/미술)〉, "
                "〈교양교과〉를 모두 한 배열에 담음)"
            ),
            "items": {
                "type": "object",
                "properties": {
                    "grade": {"type": ["integer", "null"], "description": "학년"},
                    "semester": {"type": ["integer", "null"], "description": "학기"},
                    "subject_group": {"type": ["string", "null"], "description": "교과"},
                    "subject": {"type": ["string", "null"], "description": "과목"},
                    "raw_score": {
                        "type": ["number", "null"],
                        "description": "원점수 ('원점수/과목평균' 칸의 앞 숫자)",
                    },
                    "subject_average": {
                        "type": ["number", "null"],
                        "description": "과목평균 ('원점수/과목평균' 칸의 뒤 숫자)",
                    },
                    "achievement_level": {
                        "type": ["string", "null"],
                        "description": "성취도 (A~E, P 등). 교양교과의 이수여부 P도 여기에 넣음",
                    },
                    "num_students": {
                        "type": ["integer", "null"],
                        "description": "수강자수 ('성취도(수강자수)' 칸의 괄호 안 숫자)",
                    },
                    "remarks": {"type": ["string", "null"], "description": "비고"},
                },
                "required": [
                    "grade",
                    "semester",
                    "subject_group",
                    "subject",
                    "raw_score",
                    "subject_average",
                    "achievement_level",
                    "num_students",
                    "remarks",
                ],
                "additionalProperties": False,
            },
        },
        "subject_detail_remarks": {
            "type": "array",
            "description": "5. 교과학습발달상황의 '세부능력 및 특기사항' 서술",
            "items": {
                "type": "object",
                "properties": {
                    "grade": {"type": ["integer", "null"], "description": "학년"},
                    "semester": {
                        "type": ["integer", "null"],
                        "description": "학기. 본문에 (1학기)/(2학기) 표시가 없으면 null",
                    },
                    "subject": {"type": ["string", "null"], "description": "과목"},
                    "content": {"type": "string", "description": "서술 내용 전문"},
                },
                "required": ["grade", "semester", "subject", "content"],
                "additionalProperties": False,
            },
        },
        "non_subject_records": {
            "type": "object",
            "description": "비교과 영역",
            "properties": {
                "awards": {
                    "type": "array",
                    "description": "3. 수상경력",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grade": {"type": ["integer", "null"], "description": "학년"},
                            "semester": {"type": ["integer", "null"], "description": "학기"},
                            "title": {"type": ["string", "null"], "description": "수상명"},
                            "rank": {"type": ["string", "null"], "description": "등급(위)"},
                            "date": {"type": ["string", "null"], "description": "수상연월일"},
                            "issuer": {"type": ["string", "null"], "description": "수여기관"},
                            "participants": {
                                "type": ["string", "null"],
                                "description": "참가대상(참가인원)",
                            },
                        },
                        "required": [
                            "grade",
                            "semester",
                            "title",
                            "rank",
                            "date",
                            "issuer",
                            "participants",
                        ],
                        "additionalProperties": False,
                    },
                },
                "creative_activities": {
                    "type": "array",
                    "description": "4. 창의적 체험활동상황 (자율활동/동아리활동/진로활동)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grade": {"type": ["integer", "null"], "description": "학년"},
                            "area": {"type": ["string", "null"], "description": "영역"},
                            "hours": {"type": ["integer", "null"], "description": "시간"},
                            "desired_field": {
                                "type": ["string", "null"],
                                "description": "희망분야 (진로활동에만 있음)",
                            },
                            "content": {"type": "string", "description": "특기사항 전문"},
                        },
                        "required": ["grade", "area", "hours", "desired_field", "content"],
                        "additionalProperties": False,
                    },
                },
                "volunteer_activities": {
                    "type": "array",
                    "description": "봉사활동실적",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grade": {"type": ["integer", "null"], "description": "학년"},
                            "date_or_period": {
                                "type": ["string", "null"],
                                "description": "일자 또는 기간",
                            },
                            "place": {
                                "type": ["string", "null"],
                                "description": "장소 또는 주관기관명",
                            },
                            "content": {"type": ["string", "null"], "description": "활동내용"},
                            "hours": {"type": ["integer", "null"], "description": "시간"},
                            "cumulative_hours": {
                                "type": ["integer", "null"],
                                "description": "누계시간",
                            },
                        },
                        "required": [
                            "grade",
                            "date_or_period",
                            "place",
                            "content",
                            "hours",
                            "cumulative_hours",
                        ],
                        "additionalProperties": False,
                    },
                },
                "free_semester_activities": {
                    "type": "array",
                    "description": "6. 자유학기활동상황",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grade": {"type": ["integer", "null"], "description": "학년"},
                            "semester": {"type": ["integer", "null"], "description": "학기"},
                            "area": {"type": ["string", "null"], "description": "영역"},
                            "hours": {"type": ["integer", "null"], "description": "시간"},
                            "content": {"type": "string", "description": "특기사항 전문"},
                        },
                        "required": ["grade", "semester", "area", "hours", "content"],
                        "additionalProperties": False,
                    },
                },
                "reading_activities": {
                    "type": "array",
                    "description": "7. 독서활동상황",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grade": {"type": ["integer", "null"], "description": "학년"},
                            "subject_or_area": {
                                "type": ["string", "null"],
                                "description": "과목 또는 영역",
                            },
                            "books": {
                                "type": "array",
                                "description": "독서 활동 상황 칸의 도서 목록 (원문 표기 그대로)",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["grade", "subject_or_area", "books"],
                        "additionalProperties": False,
                    },
                },
                "behavior_comments": {
                    "type": "array",
                    "description": "8. 행동특성 및 종합의견",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grade": {"type": ["integer", "null"], "description": "학년"},
                            "content": {"type": "string", "description": "의견 전문"},
                        },
                        "required": ["grade", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "awards",
                "creative_activities",
                "volunteer_activities",
                "free_semester_activities",
                "reading_activities",
                "behavior_comments",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["student", "attendance", "academic_records", "subject_detail_remarks", "non_subject_records"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """\
너는 한국 중학교 '학교생활세부사항기록부(학교생활기록부II)'를 OCR한 원문에서
정보를 뽑아 정해진 JSON 스키마로 정리하는 도구다.

원칙:
- 오직 주어진 원문에 실제로 적힌 내용만 사용한다. 추론·보완·요약·창작을 절대 하지 않는다.
- 원문에 없거나 빈 칸이면 null(또는 빈 배열)로 둔다. 그럴듯한 값을 지어내지 마라.
- 서술형 항목(특기사항, 세부능력 및 특기사항, 행동특성 및 종합의견 등)은 요약하지 말고
  원문 문장을 그대로 옮긴다. OCR 오탈자도 고치지 말고 그대로 둔다.
- 표는 <table> HTML로 주어진다. rowspan/colspan 때문에 셀이 밀려 있을 수 있으니
  머리글 행(예: 학년|수업일수|결석일수|지각|조퇴|결과|특기사항)을 기준으로 열을 맞춘다.
- 표의 '.'은 값이 0이라는 뜻이므로 숫자 항목에서는 0으로 적는다.
- 한 항목이 여러 페이지에 걸쳐 이어지면(페이지 구분자 뒤에 같은 표가 다시 나오면)
  하나의 항목으로 이어 붙인다. 특히 학년/학기 셀이 rowspan으로 앞 페이지에만 있는 경우,
  이어지는 행도 같은 학년/학기로 본다.
- 페이지 상단·하단에 나오는 '1. 2. 3. ...' 같은 눈금 숫자, 쪽번호, 섹션 제목,
  '<|det|>...<|/det|>' 좌표 표시는 데이터가 아니므로 무시한다.

필드 매핑:
- student: 인적사항(성명/성별/주민등록번호/주소)과 학적사항 이력.
- attendance: 출결상황 표. absence=결석일수, late=지각, early_leave=조퇴, result=결과,
  각각 illness=질병, unexcused=미인정, other=기타.
- academic_records: 5. 교과학습발달상황의 성적 표 전부(공통교과, 체육·예술, 교양교과).
  '원점수/과목평균'은 raw_score와 subject_average로 나누고,
  '성취도(수강자수)'는 achievement_level과 num_students로 나눈다.
  자유학기라 점수가 없고 성취도가 P인 행도 그대로 넣는다.
  교양교과의 '이수여부'는 achievement_level에, '이수시간'은 remarks에 '이수시간 17'처럼 적는다.
  학년은 표 위의 [1학년]/[2학년]/[3학년] 머리말로 판단한다.
- subject_detail_remarks: '세부능력 및 특기사항' 표. 한 셀에 여러 학기가 (1학기).../(2학기)...
  형태로 붙어 있으면 학기별로 나누어 각각 하나의 항목으로 만든다. 과목명은 셀 앞의
  '국어(자유학기):'처럼 적힌 이름에서 딴다. 내용이 '해당 사항 없음'뿐이면 항목을 만들지 않는다.
- non_subject_records.creative_activities: 자율활동/동아리활동/진로활동. 진로활동의 '희망분야'는
  desired_field에, 나머지 서술은 content에 넣는다.
- non_subject_records.free_semester_activities: 6. 자유학기활동상황. 영역/시간 칸이 비어 있고
  앞 페이지 내용이 이어지는 행은 앞 항목의 content에 이어 붙인다.
- non_subject_records.reading_activities: 도서는 '(2학기) 제목(저자), 제목(저자)' 형태이므로
  '(2학기) 제목(저자)'처럼 학기 표시를 포함해 한 권씩 문자열로 나눈다.
"""

USER_PROMPT = """\
아래는 학교생활세부사항기록부 PDF를 페이지 단위로 OCR한 원문이다.
(표는 <table> HTML, '===== Page N =====' 는 페이지 구분자)

이 원문에서만 정보를 뽑아 주어진 JSON 스키마로 정리해라.

---- OCR 원문 시작 ----
{text}
---- OCR 원문 끝 ----
"""


def build_messages(ocr_text: str, inline_schema: bool = False) -> list[dict]:
    """스키마를 서버가 강제하지 못하는 경우(inline_schema=True) 스키마를 프롬프트에 함께 넣는다."""
    system = SYSTEM_PROMPT
    if inline_schema:
        system += (
            "\n출력은 아래 JSON 스키마를 정확히 따르는 JSON 하나여야 한다. "
            "설명이나 코드펜스 없이 JSON만 출력해라.\n\n"
            + json.dumps(SCHEMA, ensure_ascii=False, indent=2)
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT.format(text=ocr_text)},
    ]


def extract(
    ocr_text: str,
    model: str,
    max_output_tokens: int,
    base_url: str,
    api_key: str,
    response_format: str,
) -> dict:
    """Send the OCR text to the model and return the parsed JSON object."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    # 추론 모델(gpt-5, o시리즈)은 temperature를 받지 않으므로 일반 모델에만 넘긴다.
    options = {} if model.startswith(("gpt-5", "o1", "o3", "o4")) else {"temperature": 0}
    if response_format == "json_schema":
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "school_record", "strict": True, "schema": SCHEMA},
        }
    elif response_format == "json_object":
        options["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(ocr_text, inline_schema=response_format != "json_schema"),
        max_completion_tokens=max_output_tokens,
        **options,
    )
    choice = response.choices[0]
    if choice.message.refusal:
        sys.exit(f"error: 모델이 응답을 거부했습니다: {choice.message.refusal}")
    if choice.finish_reason == "length":
        sys.exit(
            "error: 출력이 max-output-tokens에서 잘렸습니다. "
            "--max-output-tokens 를 늘리거나 출력 한도가 큰 모델을 쓰세요."
        )
    usage = response.usage
    if usage:
        print(
            f"토큰 사용량: 입력 {usage.prompt_tokens:,} / 출력 {usage.completion_tokens:,}",
            file=sys.stderr,
        )
    return _parse_json(choice.message.content)


def _parse_json(content: str) -> dict:
    """모델 응답에서 JSON 객체를 꺼낸다 (코드펜스나 앞뒤 설명이 섞인 경우 대비)."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.split("\n", 1)[1] if text.lstrip().startswith("json") else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            sys.exit("error: 응답에서 JSON을 찾을 수 없습니다.")
        return json.loads(text[start : end + 1])


def summarize(data: dict) -> str:
    non_subject = data.get("non_subject_records", {})
    counts = {
        "출결(학년)": len(data.get("attendance", [])),
        "교과 성적": len(data.get("academic_records", [])),
        "세부능력및특기사항": len(data.get("subject_detail_remarks", [])),
        "수상경력": len(non_subject.get("awards", [])),
        "창의적체험활동": len(non_subject.get("creative_activities", [])),
        "봉사활동": len(non_subject.get("volunteer_activities", [])),
        "자유학기활동": len(non_subject.get("free_semester_activities", [])),
        "독서활동": len(non_subject.get("reading_activities", [])),
        "행동특성": len(non_subject.get("behavior_comments", [])),
    }
    return "\n".join(f"  {label}: {n}건" for label, n in counts.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="document_ocr.txt", help="OCR 원문 텍스트 파일")
    parser.add_argument("--output", default="document_extracted.json", help="결과 JSON 경로")
    parser.add_argument("--model", default=config.LLM_MODEL, help="모델 이름 (기본값: config.py)")
    parser.add_argument(
        "--base-url",
        default=config.LLM_BASE_URL,
        help="OpenAI 호환 API 베이스 URL (기본값: config.py)",
    )
    parser.add_argument(
        "--api-key", default=config.LLM_API_KEY, help="API 키 (기본값: config.py)"
    )
    parser.add_argument(
        "--response-format",
        choices=["json_schema", "json_object", "none"],
        default=config.LLM_RESPONSE_FORMAT,
        help="구조화 출력 방식. 서버가 스키마 강제를 지원하지 않으면 json_object/none",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=config.LLM_MAX_OUTPUT_TOKENS,
        help="응답 최대 토큰 수 (서술형이 길어 넉넉히 필요함)",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="API를 호출하지 않고 전송할 프롬프트만 출력 (점검용)",
    )
    args = parser.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            ocr_text = f.read()
    except OSError as e:
        sys.exit(f"error: 입력 파일을 읽을 수 없습니다: {e}")
    if not ocr_text.strip():
        sys.exit(f"error: 입력 파일이 비어 있습니다: {args.input}")

    if args.print_prompt:
        for message in build_messages(ocr_text):
            print(f"===== {message['role']} =====")
            print(message["content"])
        return

    if not args.base_url:
        sys.exit("error: LLM 베이스 URL이 비어 있습니다. config.py 의 LLM_BASE_URL 을 설정하세요.")
    if not args.api_key:
        sys.exit(
            "error: API 키가 비어 있습니다. config.py 의 LLM_API_KEY 를 설정하거나\n"
            "  export LLM_API_KEY=... 로 지정하세요 (인증 없는 자체 서버는 아무 값이나 가능)."
        )

    print(
        f"{args.input} ({len(ocr_text):,}자) -> {args.base_url} / {args.model} 으로 구조화 추출 중...",
        file=sys.stderr,
    )
    data = extract(
        ocr_text,
        args.model,
        args.max_output_tokens,
        args.base_url,
        args.api_key,
        args.response_format,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    student = data.get("student", {}) or {}
    print(f"저장: {args.output}")
    print(f"  학생: {student.get('name')} ({student.get('gender')})")
    print(summarize(data))


if __name__ == "__main__":
    main()

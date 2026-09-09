"""PDF 한 개를 넣으면 OCR -> 구조화 추출까지 한 번에 돌리는 파이프라인.

기존 스크립트를 그대로 서브프로세스로 실행하기만 한다(로직 중복 없음):

    1) ocr.py        : PDF 각 페이지를 Unlimited-OCR로 읽어 텍스트로 (stdout -> --ocr-text)
    2) extract.py    : 그 텍스트의 표를 서식에 맞춰 JSON으로 정리 (-> --output)
    3) reconstruct.py: (선택) OCR 결과로 레이아웃 재현 HTML/PDF 생성

    python pipeline.py document.pdf
    python pipeline.py --skip-ocr            # 이미 만들어둔 OCR 텍스트 재사용
    python pipeline.py --skip-ocr --verbose  # 어떤 표를 어느 서식으로 읽었는지 확인

추출 단계는 LLM을 쓰지 않으므로 서버 설정이 필요 없고 1초도 걸리지 않는다.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def script(name: str) -> str:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        sys.exit(f"error: {name} 을 찾을 수 없습니다: {path}")
    return path


def run(step: str, cmd: list[str], stdout=None) -> None:
    """서브프로세스를 실행하고 실패하면 파이프라인을 중단한다. stderr는 그대로 흘려보낸다."""
    print(f"\n[{step}] $ {' '.join(cmd)}", file=sys.stderr, flush=True)
    started = time.monotonic()
    result = subprocess.run(cmd, stdout=stdout)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        sys.exit(f"error: [{step}] 실패 (exit {result.returncode})")
    print(f"[{step}] 완료 ({elapsed:.1f}s)", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", nargs="?", default="document.pdf", help="입력 PDF")
    parser.add_argument("--ocr-text", default="document_ocr.txt", help="OCR 텍스트 저장 경로")
    parser.add_argument("--output", default="document_extracted.json", help="결과 JSON 경로")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="OCR을 건너뛰고 기존 --ocr-text 파일을 그대로 사용",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="추출 후 reconstruct.py 로 레이아웃 재현 HTML/PDF 도 생성",
    )
    parser.add_argument("--python", default=sys.executable, help="사용할 파이썬 실행 파일")

    ocr_group = parser.add_argument_group("ocr.py 옵션 (지정한 것만 전달)")
    ocr_group.add_argument("--dpi", type=int)
    ocr_group.add_argument("--ocr-model")
    ocr_group.add_argument("--max-tokens", type=int)
    ocr_group.add_argument("--header-frac", type=float)
    ocr_group.add_argument("--footer-frac", type=float)

    extract_group = parser.add_argument_group("extract.py 옵션")
    extract_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="어떤 표를 어느 서식으로 읽었는지 표별로 출력",
    )

    args = parser.parse_args()

    # 1) OCR
    if args.skip_ocr:
        if not os.path.exists(args.ocr_text):
            sys.exit(f"error: --skip-ocr 인데 OCR 텍스트가 없습니다: {args.ocr_text}")
        print(f"[1/2] OCR 건너뜀, 기존 {args.ocr_text} 사용", file=sys.stderr)
    else:
        if not os.path.exists(args.pdf):
            sys.exit(f"error: PDF를 찾을 수 없습니다: {args.pdf}")
        cmd = [args.python, script("ocr.py"), args.pdf]
        for flag, value in [
            ("--dpi", args.dpi),
            ("--model", args.ocr_model),
            ("--max-tokens", args.max_tokens),
            ("--header-frac", args.header_frac),
            ("--footer-frac", args.footer_frac),
        ]:
            if value is not None:
                cmd += [flag, str(value)]
        # 도중에 실패해도 기존 OCR 결과를 덮어쓰지 않도록 임시 파일에 받아 성공 후 옮긴다.
        tmp = args.ocr_text + ".partial"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                run("1/2 OCR", cmd, stdout=f)
        except SystemExit:
            print(f"  (부분 결과는 {tmp} 에 남겨둡니다)", file=sys.stderr)
            raise
        shutil.move(tmp, args.ocr_text)
        size = os.path.getsize(args.ocr_text)
        print(f"[1/2 OCR] -> {args.ocr_text} ({size:,} bytes)", file=sys.stderr)

    # 2) 구조화 추출
    cmd = [
        args.python,
        script("extract.py"),
        "--input",
        args.ocr_text,
        "--output",
        args.output,
    ]
    if args.verbose:
        cmd.append("--verbose")
    run("2/2 추출", cmd)

    # 3) (선택) 레이아웃 재현
    if args.reconstruct:
        run(
            "3/3 재현",
            [args.python, script("reconstruct.py"), args.pdf, "--ocr-text", args.ocr_text],
        )

    print(f"\n완료: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

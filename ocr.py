"""Parse a PDF with Unlimited-OCR (llama.cpp, CUDA) and print the extracted text.

Renders each page of the PDF to an image, runs the OCR model on it with
the "document parsing." task (which yields structured markdown, including
tables), and prints the result page by page.

Tokens containing Han characters are banned from the sampler by default (see
`han_token_ids`), which is what keeps a Korean page from collapsing into
Chinese mid-generation.

Generation is streamed so that a page which starts repeating itself can be cut
off as soon as the repetition is visible (see `repeating_tail`) and re-read at
a temperature that can escape the loop, instead of spending the whole token
budget on it.

The model runs behind a local `llama-server` process (started and stopped by
this script) so the GGUF weights are loaded once and reused across pages.
Generation goes through llama.cpp's raw `/completion` endpoint rather than the
OpenAI-compatible chat endpoint: Unlimited-OCR's chat template has no real
turn structure (it just concatenates message content), and the server's
oaicompat message-flattening inserts a newline between the image marker and
the task text that the model was not trained on, which makes it stop
immediately without producing anything.
"""

import argparse
import atexit
import base64
import contextlib
import io
import json
import os
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pymupdf as fitz
from PIL import Image, ImageDraw

LLAMA_SERVER_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODEL_PATH = os.path.expanduser("~/llama.cpp/models/unlimited-ocr/Unlimited-OCR-BF16.gguf")
MMPROJ_PATH = os.path.expanduser(
    "~/llama.cpp/models/unlimited-ocr/mmproj-Unlimited-OCR-F16.gguf"
)
MEDIA_MARKER = "<__media__>"

# Blocking the Chinese half of the vocabulary is what keeps this model reading
# Korean. Left alone it can slip into Chinese-token space mid-page (page 12 of
# the sample: 학기|교과|과목 came out as 釣り|豆群|群号) and it never comes back:
# once there it burns the whole token budget repeating one ideograph. The model
# is not choosing a language, it is following the likeliest token, so simply
# taking those tokens away restores the correct Korean reading of the same
# image - same prompt, same speed.

# GGUF metadata value types (ggml/docs/gguf.md), used to walk the KV section.
GGUF_STRING, GGUF_ARRAY = 8, 9
GGUF_FIXED_TYPES = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F))
# Lead bytes of the UTF-8 encodings covering those ranges: a Han character could
# otherwise still be spelled out one raw byte at a time.
CJK_LEAD_BYTES = frozenset(range(0xE4, 0xEA))

# A page that starts repeating itself never recovers: at temperature 0 the
# context that produced the loop reproduces it token for token, so the page
# spends the entire --max-tokens budget on it. In the sample document that was
# pages 7, 9 and 12 at 8,192 tokens each - two thirds of the whole document's
# decoding time, and page 12 lost the grade rows the loop crowded out.
# Generating as a stream is what makes the loop visible in time to cut it off.
LOOP_WINDOW = 2400  # chars of the tail examined for a repeating block
LOOP_MAX_PERIOD = 400  # longest repeating unit to look for
LOOP_MIN_SPAN = 800  # shorter runs stay: a blank form section repeats rows too
LOOP_MIN_REPEATS = 3
LOOP_CHECK_EVERY = 50  # tokens between checks
LOOP_RETRY_TEMPERATURE = 0.3  # enough randomness to not fall into the same loop


class _Reader:
    """Sequential struct reader over the head of a GGUF file."""

    def __init__(self, stream):
        self.stream = stream

    def take(self, size: int) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("GGUF 파일이 중간에 끊겼습니다")
        return data

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> bytes:
        return self.take(self.scalar("<Q"))

    def value(self, kind: int):
        """Read one metadata value, returning it only when it is a string array."""
        if kind == GGUF_STRING:
            return self.string()
        if kind == GGUF_ARRAY:
            item_kind, count = self.scalar("<I"), self.scalar("<Q")
            if item_kind == GGUF_STRING:
                return [self.string() for _ in range(count)]
            if item_kind == GGUF_ARRAY:
                return [self.value(GGUF_ARRAY) for _ in range(count)]
            self.take(struct.calcsize(GGUF_FIXED_TYPES[item_kind]) * count)
            return None
        return self.scalar(GGUF_FIXED_TYPES[kind])


def read_gguf_vocab(path: str) -> list[bytes]:
    """Return tokenizer.ggml.tokens from a GGUF file, in token id order.

    Only the metadata at the head of the file is read, so the size of the
    weights does not matter.
    """
    with open(path, "rb") as f:
        reader = _Reader(f)
        if reader.take(4) != b"GGUF":
            raise ValueError(f"GGUF 파일이 아닙니다: {path}")
        reader.scalar("<I")  # version
        reader.scalar("<Q")  # tensor count
        for _ in range(reader.scalar("<Q")):  # kv count
            key = reader.string()
            value = reader.value(reader.scalar("<I"))
            if key == b"tokenizer.ggml.tokens":
                return value
    raise ValueError(f"tokenizer.ggml.tokens 가 없습니다: {path}")


def byte_decoder() -> dict[str, int]:
    """Reverse of GPT-2's bytes-to-unicode map, which DeepSeek's BPE vocab uses.

    Token strings are stored byte-mapped, so decoding them as plain UTF-8 finds
    no Han at all - every ideograph looks like 'ä½ ' until this is undone.
    """
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapped, extra = list(printable), 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, mapped)}


def han_token_ids(model_path: str) -> list[int]:
    """Token ids whose text contains a Han character (plus the CJK lead bytes)."""
    decoder = byte_decoder()
    banned = []
    for token_id, token in enumerate(read_gguf_vocab(model_path)):
        try:
            raw = bytes(decoder[c] for c in token.decode("utf-8"))
        except (KeyError, UnicodeDecodeError):
            continue
        if len(raw) == 1 and raw[0] in CJK_LEAD_BYTES:
            banned.append(token_id)
            continue
        text = raw.decode("utf-8", errors="replace")
        if any(low <= ord(c) <= high for c in text for low, high in HAN_RANGES):
            banned.append(token_id)
    return banned



def render_pages(
    pdf_path: str, dpi: int, header_frac: float = 0.0, footer_frac: float = 1.0
) -> list[Image.Image]:
    """Render each PDF page to an image, whiting out the top `header_frac`
    and bottom `1 - footer_frac` of the page (fixed stamp/QR/barcode areas
    that shouldn't be sent to the OCR model)."""
    zoom = dpi / 72
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")
            if header_frac > 0 or footer_frac < 1:
                w, h = img.size
                draw = ImageDraw.Draw(img)
                if header_frac > 0:
                    draw.rectangle([0, 0, w, int(h * header_frac)], fill="white")
                if footer_frac < 1:
                    draw.rectangle([0, int(h * footer_frac), w, h], fill="white")
            pages.append(img)
    return pages


def image_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def start_llama_server(args) -> subprocess.Popen:
    """Launch llama-server with the OCR model, pinned to a single CUDA device."""
    for label, path in [("--model-path", args.model_path), ("--mmproj-path", args.mmproj_path)]:
        if not os.path.exists(path):
            sys.exit(
                f"error: {label} 파일이 없습니다: {path}\n"
                "  huggingface_hub의 'hf' CLI로 받는다:\n"
                "  hf download sahilchachra/Unlimited-OCR-GGUF "
                "Unlimited-OCR-BF16.gguf mmproj-Unlimited-OCR-F16.gguf "
                f"--local-dir {os.path.dirname(path)}"
            )
    if not os.path.exists(args.llama_server_bin):
        sys.exit(f"error: llama-server 실행 파일이 없습니다: {args.llama_server_bin}")

    # --no-jinja forces llama.cpp's built-in "deepseek-ocr" template (plain
    # content concatenation) instead of the model's trivial jinja template,
    # which mishandles image markers in multi-part messages. --special makes
    # structure tokens (<|det|>, <|/det|>, <|ref|>, ...) render as literal
    # text in the output instead of being silently dropped.
    cmd = [
        args.llama_server_bin,
        "-m", args.model_path,
        "--mmproj", args.mmproj_path,
        "--host", args.host,
        "--port", str(args.port),
        "-ngl", args.n_gpu_layers,
        "-c", str(args.ctx_size),
        "-fa", args.flash_attn,
        "--chat-template", "deepseek-ocr",
        "--no-jinja",
        "--special",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Pin the media marker so we don't need to query /props for it (the server
    # otherwise generates a random one per run).
    env["LLAMA_MEDIA_MARKER"] = MEDIA_MARKER

    print(f"[launching llama-server on GPU {args.gpu_id}] $ {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stderr, stderr=subprocess.STDOUT)
    atexit.register(stop_llama_server, proc)
    return proc


def stop_llama_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def wait_for_server(base_url: str, proc: subprocess.Popen | None, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"{base_url}/health"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            sys.exit(f"error: llama-server가 시작 중 종료됨 (exit {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    sys.exit(f"error: llama-server가 {timeout:.0f}초 안에 준비되지 않았습니다: {url}")


def fetch_props(base_url: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/props", timeout=10) as resp:
        return json.loads(resp.read())


def repeating_tail(text: str) -> tuple[int, int] | None:
    """Find a block at the end of `text` that has become its own continuation.

    Returns the (start, period) of the repeated run, or None. A run has to be
    both long and repetitive enough to tell it apart from the genuinely
    identical (often blank) rows a form section can contain.
    """
    tail = text[-LOOP_WINDOW:]
    n = len(tail)
    for period in range(1, min(LOOP_MAX_PERIOD, n // LOOP_MIN_REPEATS) + 1):
        block = tail[n - period :]
        start = n - period
        while start >= period and tail[start - period : start] == block:
            start -= period
        span = n - start
        if span // period >= LOOP_MIN_REPEATS and span >= LOOP_MIN_SPAN:
            return len(text) - span, period
    return None


def run_completion(
    base_url: str,
    marker: str,
    eos_token: str,
    prompt: str,
    image_b64: str,
    max_tokens: int,
    temperature: float,
    logit_bias: list,
) -> tuple[str, bool]:
    """Read one page image, returning its text and whether a loop cut it short."""
    payload = {
        "prompt": {
            "prompt_string": marker + prompt,
            "multimodal_data": [image_b64],
        },
        "n_predict": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if logit_bias:
        # llama-server bans a token outright when its bias is false.
        payload["logit_bias"] = logit_bias
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    parts, loop, tokens = [], None, 0
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for line in resp:
                if not line.startswith(b"data: "):
                    continue
                chunk = json.loads(line[6:])
                parts.append(chunk["content"])
                tokens += 1
                if chunk.get("stop"):
                    break
                if tokens % LOOP_CHECK_EVERY == 0:
                    loop = repeating_tail("".join(parts))
                    if loop:
                        # Leaving the response unread drops the connection,
                        # which is how llama-server hears to stop generating.
                        break
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        sys.exit(f"error: llama-server 요청 실패 ({e.code}): {detail}")
    text = "".join(parts)
    if loop:
        start, period = loop
        # Keep one copy of the block: the first pass through it is usually real.
        text = text[: start + period]
        print(
            f"  [반복 감지: {tokens}토큰에서 생성 중단 "
            f"(같은 {period}자 블록 반복), 최대 {max_tokens}토큰 중 절약]",
            file=sys.stderr,
        )
    # --special (needed so <|det|>/<|ref|>/... render as text) also renders
    # the EOS token itself when generation stops naturally; strip it back off.
    if eos_token and text.endswith(eos_token):
        text = text[: -len(eos_token)]
    return text, loop is not None


def serve_forever(args) -> None:
    """Start llama-server, report readiness, and stay up until terminated.

    Loading the 5.9GB GGUF takes long enough that a batch of documents should
    not pay for it once per document. `--serve` keeps one loaded server that
    other `ocr.py` runs share through `--server-url`.
    """
    # Without this, a SIGTERM (which is how server.py stops a worker) kills this
    # process outright and leaves llama-server behind holding the GPU. Exiting
    # from the handler instead runs the atexit/finally cleanup below.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    proc = start_llama_server(args)
    base_url = f"http://{args.host}:{args.port}"
    print("[waiting for llama-server to load the model]", file=sys.stderr)
    wait_for_server(base_url, proc, args.server_timeout)
    print(f"[llama-server ready at {base_url}]", file=sys.stderr, flush=True)
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        with contextlib.suppress(Exception):
            stop_llama_server(proc)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdf", nargs="?", default="document.pdf", help="Input PDF path")
    parser.add_argument("--prompt", default="document parsing.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--ban-han",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="한자가 든 토큰을 아예 못 고르게 막는다 (기본 켬). "
             "성명 등을 한자로 병기하는 서식이면 --no-ban-han",
    )
    parser.add_argument(
        "--header-frac",
        type=float,
        default=0.115,
        help="Fraction of page height to blank out from the top (fixed stamp/QR area). 0 disables.",
    )
    parser.add_argument(
        "--footer-frac",
        type=float,
        default=0.82,
        help="Fraction of page height above which content is kept; below it is blanked out (fixed signature/barcode area). 1 disables.",
    )

    server_group = parser.add_argument_group("llama-server (CUDA)")
    server_group.add_argument("--model-path", default=MODEL_PATH, help="Unlimited-OCR GGUF 경로")
    server_group.add_argument("--mmproj-path", default=MMPROJ_PATH, help="mmproj GGUF 경로")
    server_group.add_argument("--llama-server-bin", default=LLAMA_SERVER_BIN)
    server_group.add_argument(
        "--gpu-id", default="0", help="CUDA_VISIBLE_DEVICES에 넣을 GPU 번호 (기본: 0번만 사용)"
    )
    server_group.add_argument("--n-gpu-layers", default="all", help="'all', 'auto', 또는 숫자")
    server_group.add_argument("--ctx-size", type=int, default=16384)
    server_group.add_argument("--flash-attn", choices=["on", "off", "auto"], default="off")
    server_group.add_argument("--host", default="127.0.0.1")
    server_group.add_argument("--port", type=int, default=8090)
    server_group.add_argument(
        "--server-url",
        default=None,
        help="이미 떠 있는 llama-server의 base URL (예: http://localhost:8090). "
        "지정하면 새 서버를 띄우지 않고 바로 사용한다.",
    )
    server_group.add_argument(
        "--server-timeout", type=float, default=180, help="서버 기동/모델 로딩 대기 시간(초)"
    )
    server_group.add_argument(
        "--serve",
        action="store_true",
        help="PDF를 읽지 않고 llama-server만 띄운 채 대기한다. 여러 번의 OCR 실행이 "
        "--server-url 로 이 서버를 공유하면 모델을 한 번만 올린다 (server.py가 이렇게 쓴다).",
    )
    args = parser.parse_args()

    if args.serve:
        serve_forever(args)
        return

    proc = None
    if args.server_url:
        base_url = args.server_url.rstrip("/")
        wait_for_server(base_url, None, args.server_timeout)
        marker = fetch_props(base_url)["media_marker"]
    else:
        proc = start_llama_server(args)
        base_url = f"http://{args.host}:{args.port}"
        print("[waiting for llama-server to load the model]", file=sys.stderr)
        wait_for_server(base_url, proc, args.server_timeout)
        marker = MEDIA_MARKER
    eos_token = fetch_props(base_url).get("eos_token", "")

    logit_bias = []
    if args.ban_han:
        try:
            ids = han_token_ids(args.model_path)
        except (OSError, ValueError, KeyError) as e:
            print(f"warning: 한자 토큰 목록을 읽지 못해 금지를 건너뜁니다: {e}", file=sys.stderr)
        else:
            logit_bias = [[token_id, False] for token_id in ids]
            print(f"[한자 포함 토큰 {len(ids):,}개 금지]", file=sys.stderr)

    print(f"[rendering pages from {args.pdf} at {args.dpi} dpi]", file=sys.stderr)
    pages = render_pages(args.pdf, args.dpi, args.header_frac, args.footer_frac)
    print(f"[{len(pages)} page(s)]", file=sys.stderr)

    try:
        for i, page_image in enumerate(pages, start=1):
            print(f"[OCR page {i}/{len(pages)}]", file=sys.stderr)
            image_b64 = image_to_base64_png(page_image)
            text, looped = run_completion(
                base_url,
                marker,
                eos_token,
                args.prompt,
                image_b64,
                args.max_tokens,
                args.temperature,
                logit_bias,
            )
            if looped:
                # The loop is deterministic, so re-reading the page is only
                # worth it with a temperature that can leave the cycle.
                print(
                    f"  [temperature {LOOP_RETRY_TEMPERATURE}로 이 페이지만 다시 읽음]",
                    file=sys.stderr,
                )
                retry, retry_looped = run_completion(
                    base_url,
                    marker,
                    eos_token,
                    args.prompt,
                    image_b64,
                    args.max_tokens,
                    LOOP_RETRY_TEMPERATURE,
                    logit_bias,
                )
                if not retry_looped or len(retry) > len(text):
                    text = retry
            if len(pages) > 1:
                print(f"\n\n===== Page {i} =====\n")
            print(text)
    finally:
        if proc is not None:
            with contextlib.suppress(Exception):
                stop_llama_server(proc)


if __name__ == "__main__":
    main()

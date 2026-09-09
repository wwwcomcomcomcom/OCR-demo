"""Demo web server: upload school-record PDFs, watch them parse, get JSON back.

    ./run.sh                           # 백그라운드로 띄운다 (세션이 끊겨도 유지)
    python3 server.py                  # 포그라운드: API 8081, 클라이언트 5173
    python3 server.py --gpus 0,1,3     # GPU 3장으로 PDF 3개를 동시에 처리
    python3 server.py --no-ocr         # OCR 없이(GPU 없이) UI만 띄워보기

Two ports, one process: the JSON API (--port 8081) and a plain static server
for web/ (--client-port 5173). The page finds the API through config.js, which
both servers generate from the port actually in use, so only these flags need
to change to move things around.

Nothing here re-implements the pipeline: every job shells out to the same
`ocr.py` and `extract.py` the CLI uses (see pipeline.py), so the web path and
the command line cannot drift apart.

Concurrency and where files live
--------------------------------
Each upload gets a uuid4 and its own directory under --work-dir, holding
input.pdf / ocr.txt / extracted.json. No shared filenames, so N documents can
be in flight at once without touching each other's files; the uploaded name is
kept only as a label to show in the UI.

The real limit is the GPU, not the disk. One worker thread per --gpus entry
pulls from a single queue, and each worker keeps its own `ocr.py --serve`
llama-server (one GPU each, started on that worker's first job) so the 5.9GB
model is loaded once per GPU rather than once per document. Uploads beyond the
number of GPUs queue up instead of fighting over VRAM.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
PDF_MAGIC = b"%PDF"
CHUNK = 1 << 20
LOG_TAIL = 60  # stderr lines kept per job for the UI

# ocr.py progress lines (stderr) that the UI turns into a progress bar.
RE_PAGES = re.compile(r"^\[(\d+) page\(s\)\]")
RE_PAGE = re.compile(r"^\[OCR page (\d+)/(\d+)\]")

RUNNING = ("loading", "ocr", "extract")


# ----------------------------------------------------------------- job state


@dataclasses.dataclass
class Job:
    """One uploaded PDF and everything the UI needs to know about it."""

    id: str
    filename: str
    size: int
    dir: str
    status: str = "queued"  # queued|loading|ocr|extract|done|error|cancelled
    detail: str = "대기 중"
    page: int = 0
    pages: int = 0
    gpu: str | None = None
    error: str | None = None
    created: float = dataclasses.field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    load_seconds: float | None = None  # 이 워커가 모델을 올리느라 기다린 시간 (첫 작업만)
    ocr_seconds: float | None = None
    extract_seconds: float | None = None
    log: list[str] = dataclasses.field(default_factory=list)

    @property
    def pdf_path(self) -> str:
        return os.path.join(self.dir, "input.pdf")

    @property
    def ocr_path(self) -> str:
        return os.path.join(self.dir, "ocr.txt")

    @property
    def json_path(self) -> str:
        return os.path.join(self.dir, "extracted.json")

    def public(self) -> dict:
        end = self.finished or time.time()
        return {
            "id": self.id,
            "filename": self.filename,
            "size": self.size,
            "status": self.status,
            "detail": self.detail,
            "page": self.page,
            "pages": self.pages,
            "gpu": self.gpu,
            "error": self.error,
            "created": self.created,
            "elapsed": round(end - self.started, 1) if self.started else 0.0,
            "waiting": round((self.started or time.time()) - self.created, 1),
            "load_seconds": self.load_seconds,
            "ocr_seconds": self.ocr_seconds,
            "extract_seconds": self.extract_seconds,
            "seconds_per_page": (
                round(self.ocr_seconds / self.pages, 1)
                if self.ocr_seconds and self.pages
                else None
            ),
            "has_ocr": os.path.exists(self.ocr_path),
            "log": self.log[-8:],
        }


class Store:
    """In-memory job table. The queue is the only thing workers share."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job | None] = queue.Queue()
        self.lock = threading.Lock()

    def create(self, filename: str, size: int) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = os.path.join(self.work_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        job = Job(id=job_id, filename=filename, size=size, dir=job_dir)
        with self.lock:
            self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "그런 작업이 없습니다")
        return job

    def all(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: j.created)

    def remove(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job is not None:
            shutil.rmtree(job.dir, ignore_errors=True)

    def update(self, job: Job, **fields) -> None:
        with self.lock:
            for key, value in fields.items():
                setattr(job, key, value)

    def log(self, job: Job, line: str) -> None:
        with self.lock:
            job.log.append(line)
            del job.log[:-LOG_TAIL]

    def queued_count(self) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j.status == "queued")


# -------------------------------------------------------------------- worker


class Worker(threading.Thread):
    """One GPU: keeps a warm llama-server and runs queued jobs through it."""

    def __init__(self, gpu: str, port: int, store: Store, opts: argparse.Namespace):
        super().__init__(daemon=True, name=f"worker-gpu{gpu}")
        self.gpu = gpu
        self.port = port
        self.store = store
        self.opts = opts
        self.llama: subprocess.Popen | None = None
        self.current: Job | None = None

    # -- llama-server (ocr.py --serve) ------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def ensure_llama(self, job: Job) -> None:
        """Start this worker's llama-server if it isn't up yet, and wait for it."""
        if self.opts.no_ocr:
            return
        if self.llama is not None and self.llama.poll() is None:
            return
        self.store.update(
            job, status="loading", detail=f"GPU {self.gpu} 모델 로딩 중 (최초 1회)"
        )
        started = time.monotonic()
        cmd = [
            self.opts.python,
            os.path.join(HERE, "ocr.py"),
            "--serve",
            "--gpu-id",
            self.gpu,
            "--port",
            str(self.port),
            "--server-timeout",
            str(self.opts.server_timeout),
        ]
        for flag, value in [
            ("--model-path", self.opts.model_path),
            ("--mmproj-path", self.opts.mmproj_path),
        ]:
            if value:
                cmd += [flag, value]
        log_path = os.path.join(self.store.work_dir, f"llama-gpu{self.gpu}.log")
        print(f"[gpu {self.gpu}] llama-server 기동 -> {log_path}", file=sys.stderr, flush=True)
        with open(log_path, "w", encoding="utf-8") as log_file:  # the child keeps the fd
            self.llama = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        self.wait_for_llama(log_path)
        self.store.update(job, load_seconds=round(time.monotonic() - started, 1))

    def wait_for_llama(self, log_path: str) -> None:
        deadline = time.monotonic() + self.opts.server_timeout
        while time.monotonic() < deadline:
            if self.llama.poll() is not None:
                raise RuntimeError(
                    f"GPU {self.gpu} llama-server가 시작 중 종료됨 "
                    f"(exit {self.llama.returncode}, 로그: {log_path})"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        print(f"[gpu {self.gpu}] 준비 완료", file=sys.stderr, flush=True)
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1)
        raise RuntimeError(f"GPU {self.gpu} llama-server가 시간 안에 뜨지 않았습니다")

    def stop_llama(self) -> None:
        if self.llama is None or self.llama.poll() is not None:
            return
        self.llama.terminate()
        try:
            self.llama.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.llama.kill()
            self.llama.wait()

    # -- the loop ---------------------------------------------------------

    def run(self) -> None:
        while True:
            job = self.store.queue.get()
            if job is None:  # shutdown sentinel
                break
            if job.status == "cancelled":
                continue
            self.current = job
            try:
                self.process(job)
            except Exception as e:  # a failed job must not take the worker down
                self.store.update(
                    job,
                    status="error",
                    detail="실패",
                    error=str(e),
                    finished=time.time(),
                )
                print(f"[job {job.id[:8]}] 실패: {e}", file=sys.stderr, flush=True)
            finally:
                self.current = None

    def process(self, job: Job) -> None:
        self.store.update(job, gpu=self.gpu, started=time.time())
        self.ensure_llama(job)
        self.run_ocr(job)
        self.run_extract(job)
        self.store.update(job, status="done", detail="완료", finished=time.time())

    def run_ocr(self, job: Job) -> None:
        """ocr.py PDF -> ocr.txt, following its stderr for page progress."""
        started = time.monotonic()
        if self.opts.no_ocr:
            self.store.update(job, status="ocr", detail="OCR 건너뜀 (--no-ocr)")
            source = self.opts.ocr_text
            if not source or not os.path.exists(source):
                raise RuntimeError(f"--no-ocr 인데 쓸 OCR 텍스트가 없습니다: {source}")
            shutil.copyfile(source, job.ocr_path)
            time.sleep(1)
            self.store.update(job, ocr_seconds=round(time.monotonic() - started, 1))
            return

        self.store.update(job, status="ocr", detail="OCR 준비 중", page=0, pages=0)
        cmd = [
            self.opts.python,
            os.path.join(HERE, "ocr.py"),
            job.pdf_path,
            "--server-url",
            self.base_url,
        ]
        for flag, value in [
            ("--dpi", self.opts.dpi),
            ("--max-tokens", self.opts.max_tokens),
            ("--header-frac", self.opts.header_frac),
            ("--footer-frac", self.opts.footer_frac),
            ("--model-path", self.opts.model_path),
        ]:
            if value is not None:
                cmd += [flag, str(value)]
        if not self.opts.ban_han:
            cmd.append("--no-ban-han")

        with open(job.ocr_path, "w", encoding="utf-8") as out:
            proc = subprocess.Popen(
                cmd, stdout=out, stderr=subprocess.PIPE, text=True, encoding="utf-8"
            )
            for raw in proc.stderr:
                line = raw.rstrip()
                if not line:
                    continue
                self.store.log(job, line)
                pages = RE_PAGES.match(line)
                page = RE_PAGE.match(line)
                if pages:
                    self.store.update(job, pages=int(pages.group(1)))
                elif page:
                    done, total = int(page.group(1)), int(page.group(2))
                    self.store.update(
                        job, page=done, pages=total, detail=f"{done}/{total} 페이지 OCR 중"
                    )
            code = proc.wait()
        if code != 0:
            tail = " / ".join(job.log[-3:])
            raise RuntimeError(f"OCR 실패 (exit {code}) {tail}")
        if os.path.getsize(job.ocr_path) == 0:
            raise RuntimeError("OCR 결과가 비어 있습니다")
        self.store.update(
            job,
            page=job.pages,
            detail="OCR 완료",
            ocr_seconds=round(time.monotonic() - started, 1),
        )

    def run_extract(self, job: Job) -> None:
        """extract.py ocr.txt -> extracted.json (rule-based, under a second)."""
        self.store.update(job, status="extract", detail="표 서식 매칭 중")
        started = time.monotonic()
        result = subprocess.run(
            [
                self.opts.python,
                os.path.join(HERE, "extract.py"),
                "--input",
                job.ocr_path,
                "--output",
                job.json_path,
            ],
            capture_output=True,
            text=True,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip():
                self.store.log(job, line.rstrip())
        if result.returncode != 0:
            raise RuntimeError(f"추출 실패 (exit {result.returncode}) {result.stderr.strip()}")
        self.store.update(job, extract_seconds=round(time.monotonic() - started, 2))


# ----------------------------------------------------------------------- app


def config_js(api_port: int) -> str:
    """What the page needs to reach the API: the port, not a whole URL.

    The host is whatever the browser used to get here, so this works the same
    over localhost, an IP, or a tunnel without anything to configure - as long
    as that port is what the browser can actually reach. Behind NAT/port
    forwarding where the external port differs from the one this process binds
    to, pass --public-port so this reports the external port instead.
    """
    return f"window.API_PORT = {api_port};\n"


def build_app(store: Store, workers: list[Worker], opts: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="생활기록부 파싱 데모", docs_url="/api/docs", openapi_url="/api/openapi.json")
    # The page is served from another port (5173), so every API call is
    # cross-origin. Demo server, no auth, no cookies - anyone who can reach the
    # port can call it either way, so the origin list buys nothing here.
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/config.js")
    def config():
        return Response(config_js(opts.public_port or opts.port), media_type="application/javascript")

    @app.get("/api/status")
    def status():
        return {
            "gpus": [w.gpu for w in workers],
            "busy": [w.gpu for w in workers if w.current is not None],
            "queued": store.queued_count(),
            "ocr_enabled": not opts.no_ocr,
        }

    @app.post("/api/jobs")
    def upload(files: list[UploadFile] = File(...)):
        """Accept one or many PDFs at once; each becomes its own queued job."""
        if not files:
            raise HTTPException(400, "업로드된 파일이 없습니다")
        created = []
        for upload_file in files:
            name = os.path.basename(upload_file.filename or "document.pdf")
            job = store.create(name, 0)
            try:
                size = save_pdf(upload_file, job.pdf_path, opts.max_mb)
            except HTTPException:
                store.remove(job.id)
                raise
            store.update(job, size=size)
            store.queue.put(job)
            created.append(job.public())
        return {"jobs": created}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [job.public() for job in store.all()]}

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str):
        job = store.get(job_id)
        payload = job.public()
        payload["log"] = job.log[-LOG_TAIL:]
        if job.status == "done" and os.path.exists(job.json_path):
            with open(job.json_path, encoding="utf-8") as f:
                payload["result"] = json.load(f)
        return payload

    @app.get("/api/jobs/{job_id}/json")
    def job_json(job_id: str):
        job = store.get(job_id)
        if not os.path.exists(job.json_path):
            raise HTTPException(409, "아직 결과가 없습니다")
        stem = os.path.splitext(job.filename)[0] or "document"
        return FileResponse(
            job.json_path,
            media_type="application/json",
            filename=f"{stem}_extracted.json",
        )

    @app.get("/api/jobs/{job_id}/ocr")
    def job_ocr(job_id: str):
        job = store.get(job_id)
        if not os.path.exists(job.ocr_path):
            raise HTTPException(409, "아직 OCR 결과가 없습니다")
        with open(job.ocr_path, encoding="utf-8") as f:
            return PlainTextResponse(f.read())

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = store.get(job_id)
        if job.status in RUNNING:
            raise HTTPException(409, "처리 중인 작업은 삭제할 수 없습니다")
        if job.status == "queued":
            # The worker skips it when it comes off the queue.
            store.update(job, status="cancelled", detail="취소됨", finished=time.time())
            return JSONResponse({"cancelled": True})
        store.remove(job_id)
        return JSONResponse({"deleted": True})

    # Mounted last so it only catches what the API routes above didn't. The API
    # port serves the same page as the client port, which keeps a one-port setup
    # (a tunnel, say) working.
    if os.path.isdir(WEB_DIR):
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


# ------------------------------------------------------------- client (5173)


def start_client_server(port: int, api_port: int) -> ThreadingHTTPServer:
    """Serve web/ as plain static files, with config.js generated on the fly."""

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=WEB_DIR, **kwargs)

        def do_GET(self):
            if self.path.split("?")[0] == "/config.js":
                body = config_js(api_port).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *args):  # stderr is for pipeline progress
            pass

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="client").start()
    return httpd


def save_pdf(upload_file: UploadFile, path: str, max_mb: float) -> int:
    """Stream the upload to disk, rejecting non-PDFs and oversized files."""
    limit = int(max_mb * 1024 * 1024)
    size = 0
    with open(path, "wb") as out:
        while chunk := upload_file.file.read(CHUNK):
            if size == 0 and not chunk.startswith(PDF_MAGIC):
                raise HTTPException(400, f"PDF 파일이 아닙니다: {upload_file.filename}")
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"파일이 너무 큽니다 (최대 {max_mb:g}MB)")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, f"빈 파일입니다: {upload_file.filename}")
    return size


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="0.0.0.0", help="바인딩할 주소")
    parser.add_argument("--port", type=int, default=8081, help="API 서버가 실제 바인딩할 포트")
    parser.add_argument(
        "--client-port",
        type=int,
        default=5173,
        help="web/ 을 그대로 내보내는 정적 서버 포트 (0이면 안 띄움)",
    )
    parser.add_argument(
        "--public-port",
        type=int,
        help="브라우저가 API를 부를 때 실제로 써야 하는 포트. 포트포워딩/NAT로 --port와 "
        "외부에 보이는 포트가 다를 때만 쓴다(예: 내부 8080이 공인 34762로 매핑). "
        "config.js가 이 값을 알려준다 - 비우면 --port를 그대로 쓴다.",
    )
    parser.add_argument("--pid-file", help="자기 PID를 적어둘 파일 (stop.sh 가 읽는다)")
    parser.add_argument(
        "--gpus",
        default="0",
        help="OCR에 쓸 GPU 번호를 쉼표로. 번호 하나당 워커 하나 = 동시에 처리할 문서 수 "
        "(예: --gpus 0,1,3 이면 3개 동시 처리)",
    )
    parser.add_argument(
        "--work-dir", default=os.path.join(HERE, "runs"), help="작업별 uuid 디렉터리를 만들 위치"
    )
    parser.add_argument(
        "--reset", action="store_true", help="시작할 때 --work-dir 를 비운다"
    )
    parser.add_argument("--max-mb", type=float, default=60, help="업로드 1건 크기 상한(MB)")
    parser.add_argument("--python", default=sys.executable, help="ocr.py/extract.py 실행 파이썬")
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="GPU 없이 UI만 시험한다. OCR 대신 --ocr-text 파일을 그대로 쓴다.",
    )
    parser.add_argument(
        "--ocr-text",
        default=os.path.join(HERE, "document_ocr.txt"),
        help="--no-ocr 일 때 결과 대신 쓸 기존 OCR 텍스트",
    )

    ocr_group = parser.add_argument_group("ocr.py 옵션 (지정한 것만 전달)")
    ocr_group.add_argument("--dpi", type=int)
    ocr_group.add_argument("--max-tokens", type=int)
    ocr_group.add_argument("--header-frac", type=float)
    ocr_group.add_argument("--footer-frac", type=float)
    ocr_group.add_argument("--model-path")
    ocr_group.add_argument("--mmproj-path")
    ocr_group.add_argument("--ban-han", action=argparse.BooleanOptionalAction, default=True)
    ocr_group.add_argument(
        "--llama-port-base",
        type=int,
        default=8090,
        help="GPU 워커의 llama-server 포트 시작값 (워커마다 +1)",
    )
    ocr_group.add_argument("--server-timeout", type=float, default=300, help="모델 로딩 대기(초)")

    opts = parser.parse_args()

    if opts.reset:
        shutil.rmtree(opts.work_dir, ignore_errors=True)
    os.makedirs(opts.work_dir, exist_ok=True)

    gpus = [g.strip() for g in opts.gpus.split(",") if g.strip()]
    if not gpus:
        sys.exit("error: --gpus 에 GPU 번호가 하나도 없습니다")

    store = Store(opts.work_dir)
    workers = [
        Worker(gpu, opts.llama_port_base + i, store, opts) for i, gpu in enumerate(gpus)
    ]
    for worker in workers:
        worker.start()

    app = build_app(store, workers, opts)
    client = (
        start_client_server(opts.client_port, opts.public_port or opts.port)
        if opts.client_port
        else None
    )
    if opts.pid_file:
        with open(opts.pid_file, "w", encoding="utf-8") as f:
            f.write(f"{os.getpid()}\n")

    print(
        f"\n생활기록부 파싱 데모\n"
        + (f"  화면 : http://{opts.host}:{opts.client_port}\n" if client else "")
        + f"  API  : http://{opts.host}:{opts.port} (문서 /api/docs)\n"
        f"  워커 {len(workers)}개 (GPU {', '.join(gpus)}) = 동시 처리 {len(workers)}건\n"
        f"  작업 디렉터리: {opts.work_dir}\n"
        + ("  * --no-ocr: OCR을 건너뛰고 기존 텍스트를 씁니다\n" if opts.no_ocr else ""),
        file=sys.stderr,
        flush=True,
    )
    done = threading.Event()

    def shutdown() -> None:
        """Give the GPUs back. Safe to call more than once."""
        if done.is_set():
            return
        done.set()
        print("[shutdown] 워커와 llama-server 정리 중…", file=sys.stderr, flush=True)
        if client is not None:
            with contextlib.suppress(Exception):
                client.shutdown()
        for _ in workers:
            store.queue.put(None)
        for worker in workers:
            with contextlib.suppress(Exception):
                worker.stop_llama()
        if opts.pid_file:
            with contextlib.suppress(OSError):
                os.remove(opts.pid_file)

    def on_signal(signum, _frame) -> None:
        # uvicorn은 SIGTERM/SIGINT를 자기가 받아 graceful shutdown을 한 뒤, 원래
        # 핸들러(=여기)를 되살려 같은 시그널을 다시 쏘고 죽는다. 그래서
        # uvicorn.run() 은 반환되지 않고, 뒤에 붙인 정리 코드는 영영 실행되지
        # 않는다. 정리는 이 핸들러에서 해야 llama-server가 GPU를 물고 남지 않는다.
        shutdown()
        signal.signal(signum, signal.SIG_DFL)
        signal.raise_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)

    try:
        uvicorn.run(app, host=opts.host, port=opts.port, log_level="warning")
    finally:
        shutdown()


if __name__ == "__main__":
    main()

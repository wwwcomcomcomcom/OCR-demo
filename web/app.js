/* 생활기록부 파싱 데모 프론트엔드.
 *
 * 서버는 업로드마다 uuid 작업을 만들고 GPU 워커가 순서대로 처리한다.
 * 여기서는 1.2초마다 /api/jobs 를 폴링해 진행률을 그리고, 끝난 작업은
 * /api/jobs/{id} 로 결과 JSON을 받아 표로 렌더링한다. */

/* API는 다른 포트(기본 8081)에 있다. 포트는 서버가 만들어 주는 config.js 의
 * window.API_PORT 로 오고, 호스트는 이 페이지를 연 주소를 그대로 쓴다.
 * ?api=http://... 로 직접 덮어쓸 수도 있다. */
const API = (
  new URLSearchParams(location.search).get("api") ||
  `${location.protocol}//${location.hostname}:${window.API_PORT || 8081}`
).replace(/\/$/, "");

const POLL_MS = 1200;
const RUNNING = ["queued", "loading", "ocr", "extract"];
const STATUS_TEXT = {
  queued: "대기",
  loading: "모델 로딩",
  ocr: "OCR",
  extract: "추출",
  done: "완료",
  error: "실패",
  cancelled: "취소됨",
};

const state = {
  jobs: [],
  selected: null,
  details: {}, // id -> /api/jobs/{id} 응답
  gradeFilter: null, // 교과 성적 표의 학년 필터
};

const $ = (sel) => document.querySelector(sel);

/* ------------------------------------------------------------ DOM 헬퍼 */

function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const dash = (value) =>
  value === null || value === undefined || value === "" ? "–" : String(value);

function table(headers, rows) {
  const thead = h(
    "thead",
    {},
    h("tr", {}, headers.map((head) =>
      h("th", { class: head.num ? "num" : null }, head.label ?? head)
    ))
  );
  const tbody = h(
    "tbody",
    {},
    rows.map((row) =>
      h("tr", {}, row.map((cell, i) => {
        const spec = headers[i] || {};
        const value = cell && typeof cell === "object" ? cell.value : cell;
        const cls = [spec.num ? "num" : null, spec.wrap ? "wrap" : null]
          .filter(Boolean)
          .join(" ");
        return h("td", { class: cls || null }, dash(value));
      }))
    )
  );
  return h("div", { class: "table-wrap" }, h("table", {}, thead, tbody));
}

function block(title, ...content) {
  return h("section", { class: "block" }, h("h3", { text: title }), ...content);
}

/* ------------------------------------------------------------- 업로드 */

function showUploadError(message) {
  const box = $("#upload-error");
  box.textContent = message;
  box.hidden = !message;
}

async function upload(files) {
  const pdfs = [...files].filter(
    (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
  );
  if (!pdfs.length) {
    showUploadError("PDF 파일만 올릴 수 있습니다.");
    return;
  }
  showUploadError("");
  const form = new FormData();
  pdfs.forEach((file) => form.append("files", file, file.name));
  try {
    const resp = await fetch(`${API}/api/jobs`, { method: "POST", body: form });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `업로드 실패 (${resp.status})`);
    }
    const { jobs } = await resp.json();
    if (jobs.length) state.selected = jobs[jobs.length - 1].id;
    await poll();
  } catch (err) {
    showUploadError(err.message);
  }
}

function wireDropzone() {
  const zone = $("#dropzone");
  const input = $("#file-input");
  // input.click() 이 만든 클릭도 폼까지 올라오므로, 그대로 두면 서로를 무한히 부른다.
  zone.addEventListener("click", (e) => {
    if (e.target !== input) input.click();
  });
  zone.addEventListener("submit", (e) => e.preventDefault());
  input.addEventListener("change", () => {
    upload(input.files);
    input.value = "";
  });
  for (const type of ["dragenter", "dragover"]) {
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.add("hot");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    zone.addEventListener(type, (e) => {
      e.preventDefault();
      zone.classList.remove("hot");
    });
  }
  zone.addEventListener("drop", (e) => e.dataTransfer && upload(e.dataTransfer.files));
}

/* --------------------------------------------------------------- 폴링 */

async function poll() {
  try {
    const [jobsResp, statusResp] = await Promise.all([
      fetch(`${API}/api/jobs`),
      fetch(`${API}/api/status`),
    ]);
    const { jobs } = await jobsResp.json();
    state.jobs = jobs;
    renderStatus(await statusResp.json());
    renderList();
    await refreshDetail();
  } catch {
    $("#status-pill").textContent = "서버 연결 끊김";
  }
}

async function refreshDetail() {
  const job = state.jobs.find((j) => j.id === state.selected);
  if (!job) {
    renderDetail(null);
    return;
  }
  const cached = state.details[job.id];
  const needsFetch = !cached || cached.status !== job.status || RUNNING.includes(job.status);
  if (needsFetch) {
    const resp = await fetch(`${API}/api/jobs/${job.id}`);
    if (resp.ok) state.details[job.id] = await resp.json();
  }
  renderDetail(state.details[job.id] || job);
}

function select(id) {
  state.selected = id;
  state.gradeFilter = null;
  renderList();
  refreshDetail();
}

/* --------------------------------------------------------- 목록 렌더링 */

function renderStatus(status) {
  const parts = [`GPU ${status.gpus.join(", ")} (동시 ${status.gpus.length}건)`];
  if (status.busy.length) parts.push(`처리 중 ${status.busy.length}`);
  if (status.queued) parts.push(`대기 ${status.queued}`);
  if (!status.ocr_enabled) parts.push("OCR 비활성(--no-ocr)");
  $("#status-pill").textContent = parts.join(" · ");
}

function progressOf(job) {
  if (job.status === "ocr" && job.pages) return job.page / job.pages;
  if (job.status === "extract") return 1;
  return null;
}

function renderList() {
  const list = $("#job-list");
  list.textContent = "";
  $("#job-empty").hidden = state.jobs.length > 0;

  for (const job of state.jobs) {
    const ratio = progressOf(job);
    const running = RUNNING.includes(job.status);
    const card = h(
      "li",
      {
        class: `job${job.id === state.selected ? " selected" : ""}`,
        onclick: () => select(job.id),
      },
      h(
        "div",
        { class: "job-top" },
        h("span", { class: "job-name", text: job.filename, title: job.filename }),
        h("span", { class: `badge ${job.status}`, text: STATUS_TEXT[job.status] || job.status })
      ),
      h(
        "div",
        { class: "job-meta" },
        h("span", { class: "grow", text: job.detail }),
        h("span", { text: job.gpu !== null ? `GPU ${job.gpu}` : "" }),
        h("span", { text: job.elapsed ? `${job.elapsed.toFixed(0)}초` : "" })
      ),
      running
        ? h(
            "div",
            { class: `bar${ratio === null ? " indeterminate" : ""}` },
            h("i", { style: ratio === null ? "" : `width:${Math.round(ratio * 100)}%` })
          )
        : null,
      h(
        "div",
        { class: "job-actions" },
        job.status === "done"
          ? h("a", {
              class: "link-btn",
              href: `${API}/api/jobs/${job.id}/json`,
              text: "JSON 다운로드",
              onclick: (e) => e.stopPropagation(),
            })
          : null,
        !running
          ? h("button", {
              class: "link-btn",
              type: "button",
              text: job.status === "queued" ? "취소" : "삭제",
              onclick: (e) => {
                e.stopPropagation();
                removeJob(job.id);
              },
            })
          : null
      )
    );
    list.append(card);
  }
}

async function removeJob(id) {
  await fetch(`${API}/api/jobs/${id}`, { method: "DELETE" });
  delete state.details[id];
  if (state.selected === id) state.selected = null;
  poll();
}

async function clearDone() {
  const finished = state.jobs.filter((j) => !RUNNING.includes(j.status));
  await Promise.all(finished.map((j) => fetch(`${API}/api/jobs/${j.id}`, { method: "DELETE" })));
  finished.forEach((j) => delete state.details[j.id]);
  if (finished.some((j) => j.id === state.selected)) state.selected = null;
  poll();
}

/* --------------------------------------------------------- 상세 렌더링 */

/* 상세 머리말: 상태 · 크기 · GPU · 총 소요 시간, 그리고 끝났으면 단계별 시간까지.
 * (모델 로딩은 그 GPU의 첫 작업에서만 붙는다) */
function detailSub(job) {
  const parts = [STATUS_TEXT[job.status] || job.status, `${(job.size / 1048576).toFixed(1)}MB`];
  if (job.gpu !== null) parts.push(`GPU ${job.gpu}`);
  if (job.elapsed) parts.push(`총 ${job.elapsed.toFixed(1)}초`);
  if (job.load_seconds) parts.push(`모델 로딩 ${job.load_seconds}초`);
  if (job.ocr_seconds) {
    const perPage = job.seconds_per_page ? ` (${job.seconds_per_page}초/쪽)` : "";
    parts.push(`OCR ${job.ocr_seconds}초${perPage}`);
  }
  if (job.extract_seconds) parts.push(`추출 ${job.extract_seconds}초`);
  return parts.join(" · ");
}

function renderDetail(job) {
  const panel = $("#detail");
  panel.textContent = "";
  if (!job) {
    panel.append(
      h(
        "div",
        { class: "placeholder" },
        h("p", { text: "왼쪽에서 문서를 선택하면 파싱 결과가 여기에 표시됩니다." })
      )
    );
    return;
  }

  panel.append(
    h(
      "div",
      { class: "detail-head" },
      h(
        "div",
        {},
        h("h2", { text: job.filename }),
        h("div", { class: "sub", text: detailSub(job) })
      ),
      h(
        "div",
        { class: "head-actions" },
        job.status === "done"
          ? h("a", { class: "btn primary", href: `${API}/api/jobs/${job.id}/json`, text: "JSON 다운로드" })
          : null,
        job.has_ocr
          ? h("a", {
              class: "btn",
              href: `${API}/api/jobs/${job.id}/ocr`,
              target: "_blank",
              text: "OCR 원문",
            })
          : null
      )
    )
  );

  if (job.status === "error") {
    panel.append(h("div", { class: "error-box", text: job.error || "알 수 없는 오류" }));
    panel.append(block("실행 로그", h("pre", { class: "log", text: (job.log || []).join("\n") })));
    return;
  }
  if (job.status !== "done") {
    panel.append(renderProgress(job));
    return;
  }
  renderResult(panel, job);
}

function renderProgress(job) {
  const ratio = progressOf(job);
  return h(
    "div",
    {},
    h(
      "div",
      { class: "progress-panel" },
      h("div", {
        class: "big",
        text:
          job.status === "ocr" && job.pages
            ? `${job.page} / ${job.pages} 페이지`
            : STATUS_TEXT[job.status] || job.status,
      }),
      h("p", { text: job.detail }),
      h(
        "div",
        { class: `bar${ratio === null ? " indeterminate" : ""}` },
        h("i", { style: ratio === null ? "" : `width:${Math.round(ratio * 100)}%` })
      )
    ),
    job.log && job.log.length
      ? h(
          "details",
          {},
          h("summary", { text: "실행 로그" }),
          h("pre", { class: "log", text: job.log.join("\n") })
        )
      : null
  );
}

function maskRrn(value) {
  if (!value) return value;
  return value.replace(/^(\d{6}-\d)\d{6}$/, "$1******");
}

function sum(rows, pick) {
  return rows.reduce((total, row) => total + (Number(pick(row)) || 0), 0);
}

function renderResult(panel, job) {
  const data = job.result;
  if (!data) {
    panel.append(h("p", { class: "note", text: "결과를 불러오는 중…" }));
    return;
  }
  const student = data.student || {};
  const attendance = data.attendance || [];
  const records = data.academic_records || [];
  const extra = data.non_subject_records || {};

  // 요약 지표
  const scored = records.filter((r) => typeof r.raw_score === "number");
  const average = scored.length
    ? (sum(scored, (r) => r.raw_score) / scored.length).toFixed(1)
    : "–";
  panel.append(
    h(
      "div",
      { class: "stats" },
      stat(records.length, "교과 성적"),
      stat(average, "원점수 평균"),
      stat(sum(attendance, (a) => a.absence?.illness + a.absence?.unexcused + a.absence?.other), "결석(일)"),
      stat(sum(extra.volunteer_activities || [], (v) => v.hours), "봉사(시간)"),
      stat((extra.awards || []).length, "수상"),
      stat(sum(extra.creative_activities || [], (c) => c.hours), "창체(시간)")
    )
  );

  // 인적사항
  panel.append(
    block(
      "인적사항",
      h(
        "dl",
        { class: "kv" },
        kv("성명", student.name),
        kv("성별", student.gender),
        kv("주민등록번호", maskRrn(student.resident_registration_number)),
        kv("주소", student.address)
      ),
      h("p", { class: "note", text: "주민등록번호 뒷자리는 화면에서만 가립니다. 내려받는 JSON에는 원문이 들어 있습니다." })
    )
  );

  if ((student.school_history || []).length) {
    panel.append(
      block(
        "학적사항",
        table(
          [{ label: "일자" }, { label: "내용", wrap: true }],
          student.school_history.map((row) => [row.date, row.description])
        )
      )
    );
  }

  if ((student.enrollment || []).length) {
    panel.append(
      block(
        "학년별 학적",
        table(
          [
            { label: "학년", num: true },
            { label: "반", num: true },
            { label: "번호", num: true },
            { label: "담임 성명" },
          ],
          student.enrollment.map((row) => [row.grade, row.class_name, row.number, row.homeroom_teacher])
        )
      )
    );
  }

  if (attendance.length) {
    panel.append(
      block(
        "출결상황",
        table(
          [
            { label: "학년", num: true },
            { label: "수업일수", num: true },
            { label: "결석 질병", num: true },
            { label: "결석 미인정", num: true },
            { label: "결석 기타", num: true },
            { label: "지각 질병", num: true },
            { label: "지각 미인정", num: true },
            { label: "조퇴 질병", num: true },
            { label: "조퇴 미인정", num: true },
            { label: "결과", num: true },
          ],
          attendance.map((row) => [
            row.grade,
            row.school_days,
            row.absence?.illness,
            row.absence?.unexcused,
            row.absence?.other,
            row.late?.illness,
            row.late?.unexcused,
            row.early_leave?.illness,
            row.early_leave?.unexcused,
            (row.result?.illness || 0) + (row.result?.unexcused || 0) + (row.result?.other || 0),
          ])
        )
      )
    );
  }

  if (records.length) panel.append(renderRecords(records));

  const sections = [
    [
      "수상경력",
      extra.awards,
      [{ label: "학년", num: true }, { label: "수상명", wrap: true }, { label: "등급" }, { label: "수상연월일" }, { label: "수여기관" }, { label: "참가대상" }],
      (r) => [r.grade, r.title, r.rank, r.date, r.issuer, r.participants],
    ],
    [
      "창의적 체험활동",
      extra.creative_activities,
      [{ label: "학년", num: true }, { label: "영역" }, { label: "시간", num: true }, { label: "희망분야" }],
      (r) => [r.grade, r.area, r.hours, r.desired_field],
    ],
    [
      "봉사활동",
      extra.volunteer_activities,
      [{ label: "학년", num: true }, { label: "일자/기간" }, { label: "장소" }, { label: "활동내용", wrap: true }, { label: "시간", num: true }, { label: "누계", num: true }],
      (r) => [r.grade, r.date_or_period, r.place, r.content, r.hours, r.cumulative_hours],
    ],
    [
      "자유학기활동",
      extra.free_semester_activities,
      [{ label: "학년", num: true }, { label: "학기", num: true }, { label: "영역" }, { label: "프로그램", wrap: true }, { label: "시간", num: true }],
      (r) => [r.grade, r.semester, r.area, r.program, r.hours],
    ],
    [
      "독서활동",
      extra.reading_activities,
      [{ label: "학년", num: true }, { label: "학기", num: true }, { label: "과목/영역" }, { label: "도서", wrap: true }],
      (r) => [r.grade, r.semester, r.subject_or_area, (r.books || []).join(", ")],
    ],
  ];
  for (const [title, rows, headers, pick] of sections) {
    if (rows && rows.length) {
      panel.append(block(`${title} (${rows.length}건)`, table(headers, rows.map(pick))));
    }
  }

  panel.append(
    block(
      "JSON 원문",
      h(
        "details",
        {},
        h("summary", { text: "펼쳐보기" }),
        h("pre", { class: "json", text: JSON.stringify(data, null, 2) })
      )
    )
  );
}

function stat(value, label) {
  return h("div", { class: "stat" }, h("b", { text: dash(value) }), h("span", { text: label }));
}

function kv(label, value) {
  return h("div", {}, h("dt", { text: label }), h("dd", { text: dash(value) }));
}

function renderRecords(records) {
  const grades = [...new Set(records.map((r) => r.grade))].sort();
  const active = state.gradeFilter;
  const rows = records.filter((r) => active === null || r.grade === active);

  const filters = h(
    "div",
    { class: "filters" },
    h("button", {
      class: `chip${active === null ? " on" : ""}`,
      type: "button",
      text: `전체 ${records.length}`,
      onclick: () => {
        state.gradeFilter = null;
        refreshDetail();
      },
    }),
    grades.map((grade) =>
      h("button", {
        class: `chip${active === grade ? " on" : ""}`,
        type: "button",
        text: `${grade}학년`,
        onclick: () => {
          state.gradeFilter = grade;
          refreshDetail();
        },
      })
    )
  );

  return block(
    "교과학습발달상황",
    filters,
    table(
      [
        { label: "학년", num: true },
        { label: "학기", num: true },
        { label: "구분" },
        { label: "교과" },
        { label: "과목" },
        { label: "원점수", num: true },
        { label: "과목평균", num: true },
        { label: "성취도" },
        { label: "수강자수", num: true },
        { label: "이수시간", num: true },
      ],
      rows.map((r) => [
        r.grade,
        r.semester,
        r.area,
        r.subject_group,
        r.subject,
        r.raw_score,
        r.subject_average,
        r.achievement_level,
        r.num_students,
        r.completed_hours,
      ])
    )
  );
}

/* ---------------------------------------------------------------- 시작 */

wireDropzone();
$("#clear-done").addEventListener("click", clearDone);
poll();
setInterval(poll, POLL_MS);

"""Map an OCR'd 학교생활세부사항기록부 onto a fixed JSON schema, without an LLM.

`ocr.py` emits every page as text blocks plus <table> HTML. This step does not
ask a model to organise that: each table this form carries is declared below as
a TEMPLATE - a header signature plus the columns worth keeping - and a table is
read only when its header is recognised at the place it is expected.

Nothing here is hard-coded to exact cell counts, because the OCR routinely adds
an empty column, drops the rowspan cell a table continued from the previous page
with, or garbles a header into nonsense. So matching is soft on three levels:

    1. header signature   - fuzzy (difflib) label matching, not equality
    2. column alignment   - two candidate mappings per row (grid-aware and
                            positional), scored by per-column value validators;
                            the better one wins
    3. table identity     - a table whose header is unreadable is retried as a
                            continuation of the last table matched in the same
                            section, and kept only if its rows validate

The trade is deliberate: only declared, positionally anchored values come out
(서술형 특기사항 is dropped entirely), but the run is instant and repeatable.
"""

import argparse
import html as html_lib
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# ------------------------------------------------------------ OCR text parsing

PAGE_RE = re.compile(r"^=====\s*Page\s+(\d+)\s*=====\s*$", re.M)
# Each block is announced by its own <|det|>type [x0, y0, x1, y1]<|/det|> tag and
# runs until the next tag (a block's body may span several lines).
DET_RE = re.compile(r"<\|det\|>(\w*)\s*\[[^\]\n]*\]<\|/det\|>")
# Rows/cells are matched leniently: a page cut off at ocr.py's token cap can end
# without its closing </td>, </tr> or </table>.
TABLE_RE = re.compile(r"(<table[^>]*>.*?(?:</table>|$))", re.S)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)(?:</tr>|(?=<tr)|$)", re.S)
CELL_RE = re.compile(r"<td([^>]*)>(.*?)(?:</td>|(?=<td)|(?=</tr>)|$)", re.S)
SPAN_RE = re.compile(r'(colspan|rowspan)\s*=\s*"?(\d+)"?')


@dataclass
class Block:
    """One <|det|>-tagged block of the OCR output."""

    page: int
    kind: str
    text: str

    @property
    def is_table(self) -> bool:
        return "<tr" in self.text


def split_blocks(page: int, kind: str, text: str):
    """Yield one Block per table, so that a chunk holding several tables (a page
    whose <|det|> tags went missing) is not read as one big grid."""
    for piece in TABLE_RE.split(text):
        if piece.strip():
            yield Block(page, kind, piece.strip())


def iter_blocks(text: str):
    """Split the OCR output into per-page, per-block pieces, in reading order."""
    pages = PAGE_RE.split(text)
    # split() gives [preamble, page_no, body, page_no, body, ...]
    for i in range(1, len(pages) - 1, 2):
        page, body = int(pages[i]), pages[i + 1]
        marks = list(DET_RE.finditer(body))
        if not marks:
            yield from split_blocks(page, "", body)
            continue
        yield from split_blocks(page, "", body[: marks[0].start()])
        for j, mark in enumerate(marks):
            end = marks[j + 1].start() if j + 1 < len(marks) else len(body)
            yield from split_blocks(page, mark.group(1), body[mark.end() : end])


class Cell:
    """A table cell. Identity matters: one cell can occupy several grid slots."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Cell({self.text!r})"


def clean(text: str) -> str:
    """Unescape entities and squeeze whitespace out of a cell's text."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html_lib.unescape(text)).strip()


def expand_table(html: str) -> list[list[Cell | None]]:
    """Expand a <table> into a rectangular grid, honouring rowspan/colspan.

    Every slot holds the Cell object that covers it, so a cell spanning several
    columns is the *same* object in each of them - which is what lets the column
    mapping below tell "this value covers 2 of the 3 columns of this header
    group" from "this value happens to sit in one of them".
    """
    grid: list[dict[int, Cell]] = []
    carry: dict[int, list] = {}  # column -> [cell, rows still to fill]
    for raw_row in ROW_RE.findall(html):
        row: dict[int, Cell] = {}
        for col in sorted(carry):
            entry = carry[col]
            row[col] = entry[0]
            entry[1] -= 1
        carry = {col: e for col, e in carry.items() if e[1] > 0}
        col = 0
        for attrs, body in CELL_RE.findall(raw_row):
            spans = dict(SPAN_RE.findall(attrs))
            colspan = max(1, int(spans.get("colspan", 1)))
            rowspan = max(1, int(spans.get("rowspan", 1)))
            cell = Cell(clean(body))
            for _ in range(colspan):
                while col in row:
                    col += 1
                row[col] = cell
                if rowspan > 1:
                    carry[col] = [cell, rowspan - 1]
                col += 1
        grid.append(row)
    width = max((max(row) + 1 for row in grid if row), default=0)
    return [[row.get(i) for i in range(width)] for row in grid]


def row_texts(row: list[Cell | None]) -> list[str]:
    """The texts of a grid row, one slot per column ('' where nothing covers it)."""
    return [cell.text if cell else "" for cell in row]


def logical_cells(row: list[Cell | None]) -> list[str]:
    """The row's cells in order, each counted once however many slots it spans."""
    out, previous = [], None
    for cell in row:
        if cell is not None and cell is not previous:
            out.append(cell.text)
        previous = cell
    return out


def distinct_texts(rows: list[list[Cell | None]]) -> list[str]:
    """Non-empty cell texts of some rows, de-duplicated, in order."""
    out = []
    for row in rows:
        for text in logical_cells(row):
            if text and text not in out:
                out.append(text)
    return out


# ---------------------------------------------------------- fuzzy label match

PUNCT_RE = re.compile(r"[\s·．.,()\[\]{}<>〈〉《》「」/|:：;~\-_'\"’”]+")

MIN_LABEL_SIMILARITY = 0.6  # 답임성명/담임성명, 수업입안수/수업일수 정도는 붙어야 한다
MIN_HEADER_ROW_RATIO = 0.6  # 한 행을 머리글로 볼 최소 라벨 비율
MIN_TEMPLATE_SCORE = 0.6
MIN_ROW_SCORE = 0.5
TEXT_WEIGHT = 0.25  # a plausible free-text value is weak evidence, not proof


def normalize(text: str) -> str:
    return PUNCT_RE.sub("", text)


def similarity(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a and a == b else 0.0
    return SequenceMatcher(None, a, b).ratio()


def matches_label(text: str, labels) -> bool:
    return any(similarity(text, label) >= MIN_LABEL_SIMILARITY for label in labels)


# -------------------------------------------------------------- column kinds

ZERO_MARKS = {".", "·", ",", "、", "。", "，"}  # 출결표의 '.' 은 값 0 이라는 뜻


def _grade(v: str) -> bool:
    return bool(re.fullmatch(r"[1-6]", v.strip()))


def _semester(v: str) -> bool:
    return bool(re.fullmatch(r"[1-4]", v.strip()))


def _count(v: str) -> bool:
    return v.strip() in ZERO_MARKS or bool(re.fullmatch(r"\d{1,4}", v.strip()))


def _date(v: str) -> bool:
    return bool(re.search(r"\d{4}\s*[.년\-]\s*\d{1,2}", v))


def _score(v: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{1,3}(\.\d+)?\s*/\s*\d{1,3}(\.\d+)?\s*", v))


def _level(v: str) -> bool:
    return bool(re.fullmatch(r"\s*[A-EPa-ep]\s*(\(\s*\d+\s*\))?\s*", v))


# kind -> (validator, weight when it holds)
KINDS = {
    "grade": (_grade, 1.0),
    "semester": (_semester, 1.0),
    "count": (_count, 1.0),
    "date": (_date, 1.0),
    "score": (_score, 1.0),
    "level": (_level, 1.0),
    "text": (lambda v: True, TEXT_WEIGHT),
}


JUNK_RUN_RE = re.compile(r"(.)\1{9,}")
JUNK_CHARS = 300


def is_junk(value: str) -> bool:
    """A cell the OCR degenerated into (a character repeated to the token cap).

    Also catches the 서술형 blocks, which are longer than any value this schema
    keeps - the longest is a 독서활동 book list at ~100자.
    """
    return len(value) > JUNK_CHARS or bool(JUNK_RUN_RE.search(value))


def to_int(value):
    """'.' -> 0, '17시간' -> 17, '' -> None."""
    if not value:
        return None
    if value.strip() in ZERO_MARKS:
        return 0
    m = re.search(r"\d+", value)
    return int(m.group()) if m else None


def to_float(value):
    if not value:
        return None
    m = re.search(r"\d+(?:\.\d+)?", value)
    return float(m.group()) if m else None


def blank(value):
    """Normalise an empty-ish cell to None."""
    value = (value or "").strip()
    return None if not value or value in ZERO_MARKS else value


# ------------------------------------------------------------------ templates


@dataclass
class Column:
    """One column of a 표 서식."""

    name: str  # key in the mapped row
    label: str  # header text to look for (fuzzily)
    kind: str = "text"
    carry: bool = False  # inherit the last row's value when the cell is empty
    key: bool = False  # the row is dropped when every key column is empty


@dataclass
class Template:
    """A table this form is known to contain, and the columns worth keeping."""

    name: str  # 사람이 읽는 이름 (로그용)
    columns: list[Column]
    labels: list[str] = field(default_factory=list)  # header signature
    section: str | None = None  # 이 서식이 나올 수 있는 절 (없으면 아무 데나)
    # 표 안에 라벨이 따로 붙어 나오는 행(창의적체험활동의 '희망분야')은
    # 영역 칸이 앞 행에서 넘어와도 그 자체로 하나의 항목이다.
    keep_if: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.labels:
            self.labels = [c.label for c in self.columns]

    @property
    def key_names(self) -> list[str]:
        return [c.name for c in self.columns if c.key] or [self.columns[0].name]


COUNT_GROUPS = {"absence": "결석일수", "late": "지각", "early_leave": "조퇴", "result": "결과"}


def _counts(prefix: str) -> list[Column]:
    """결석일수/지각/조퇴/결과 are each split into 질병·미인정·기타."""
    label = COUNT_GROUPS[prefix]
    return [
        Column(f"{prefix}_{name}", f"{label} {sub}", "count")
        for name, sub in (("illness", "질병"), ("unexcused", "미인정"), ("other", "기타"))
    ]


TEMPLATES = [
    Template(
        name="학적(반·번호·담임)",
        section="1",
        columns=[
            Column("grade", "학년", "grade", carry=True, key=True),
            Column("class_name", "반", "count"),
            Column("number", "번호", "count"),
            Column("homeroom_teacher", "담임성명", "text", key=True),
        ],
    ),
    Template(
        name="2. 출결상황",
        section="2",
        columns=[
            Column("grade", "학년", "grade", carry=True),
            Column("school_days", "수업일수", "count", key=True),
            *_counts("absence"),
            *_counts("late"),
            *_counts("early_leave"),
            *_counts("result"),
            Column("remarks", "특기사항", "text"),
        ],
        labels=[
            "학년",
            "수업일수",
            "결석일수",
            "지각",
            "조퇴",
            "결과",
            "질병",
            "미인정",
            "기타",
            "특기사항",
        ],
    ),
    Template(
        name="3. 수상경력",
        section="3",
        columns=[
            Column("grade", "학년(학기)", "grade", carry=True),
            Column("title", "수상명", "text", key=True),
            Column("rank", "등급(위)", "text"),
            Column("date", "수상연월일", "date"),
            Column("issuer", "수여기관", "text"),
            Column("participants", "참가대상(참가인원)", "text"),
        ],
    ),
    Template(
        name="4. 창의적 체험활동상황",
        section="4",
        columns=[
            Column("grade", "학년", "grade", carry=True),
            Column("area", "영역", "text", key=True),
            Column("hours", "시간", "count"),
            Column("note", "특기사항", "text"),
        ],
        labels=["학년", "창의적체험활동", "영역", "시간", "특기사항"],
        keep_if=["희망분야"],
    ),
    Template(
        name="4. 봉사활동실적",
        section="4",
        columns=[
            Column("grade", "학년", "grade", carry=True),
            Column("date_or_period", "일자 또는 기간", "date", key=True),
            Column("place", "장소 또는 주관기관명", "text"),
            Column("content", "활동내용", "text", key=True),
            Column("hours", "시간", "count"),
            Column("cumulative_hours", "누계시간", "count"),
        ],
        labels=["학년", "봉사활동실적", "일자 또는 기간", "장소 또는 주관기관명", "활동내용", "시간"],
    ),
    Template(
        name="5. 교과학습발달상황(공통교과)",
        section="5",
        columns=[
            Column("semester", "학기", "semester", carry=True),
            Column("subject_group", "교과", "text"),
            Column("subject", "과목", "text", key=True),
            Column("score", "원점수/과목평균", "score"),
            Column("level", "성취도(수강자수)", "level"),
            Column("remarks", "비고", "text"),
        ],
    ),
    Template(
        name="5. 교과학습발달상황(체육·예술)",
        section="5",
        columns=[
            Column("semester", "학기", "semester", carry=True),
            Column("subject_group", "교과", "text"),
            Column("subject", "과목", "text", key=True),
            Column("level", "성취도", "level"),
            Column("remarks", "비고", "text"),
        ],
    ),
    Template(
        name="5. 교과학습발달상황(교양교과)",
        section="5",
        columns=[
            Column("semester", "학기", "semester", carry=True),
            Column("subject_group", "교과", "text"),
            Column("subject", "과목", "text", key=True),
            Column("completed_hours", "이수시간", "count"),
            Column("level", "이수여부", "level"),
            Column("remarks", "비고", "text"),
        ],
    ),
    Template(
        name="6. 자유학기활동상황",
        section="6",
        columns=[
            Column("grade", "학년", "grade", carry=True),
            Column("semester", "학기", "semester", carry=True),
            Column("area", "영역", "text", key=True),
            Column("hours", "시간", "count"),
            Column("note", "특기사항", "text"),
        ],
        labels=["학년", "학기", "자유학기활동상황", "영역", "시간", "특기사항"],
    ),
    Template(
        name="7. 독서활동상황",
        section="7",
        columns=[
            Column("grade", "학년", "grade", carry=True),
            Column("subject_or_area", "과목 또는 영역", "text"),
            Column("books", "독서 활동 상황", "text", key=True),
        ],
    ),
]


# --------------------------------------------------------------- table reading


def find_header(grid, template) -> tuple[int, int] | None:
    """Locate the run of header rows near the top of a table.

    A row is a header row when most of its cells read like this template's
    labels. The run may start below row 0 (page 1's 학적 table opens with a
    졸업대상번호 line above its header) and is at most two rows deep.
    """

    def score(index: int) -> float:
        texts = [t for t in distinct_texts([grid[index]]) if t]
        if not texts:
            return 0.0
        hits = sum(1 for t in texts if matches_label(t, template.labels))
        return hits / len(texts)

    for start in range(min(3, len(grid))):
        if score(start) < MIN_HEADER_ROW_RATIO:
            continue
        end = start
        while end + 1 < len(grid) and end - start < 1 and score(end + 1) >= MIN_HEADER_ROW_RATIO:
            end += 1
        return start, end
    return None


def template_score(grid, header: tuple[int, int], template) -> float:
    """How well a table's header matches this 서식 (0..1).

    Divided by the larger of the two label counts so that a header carrying
    columns the template does not know about (원점수/과목평균 in the 공통교과
    table vs the shorter 체육·예술 one) scores lower than the exact fit.
    """
    texts = distinct_texts(grid[header[0] : header[1] + 1])
    if not texts:
        return 0.0
    hits = sum(1 for label in template.labels if any(similarity(t, label) >= MIN_LABEL_SIMILARITY for t in texts))
    return hits / max(len(template.labels), len(texts))


def column_groups(grid, header: tuple[int, int]) -> list[tuple[str, tuple[int, int]]]:
    """Per grid column: its header text, and the span of the header cell over it.

    The span is what disambiguates a shifted row: 수상경력's 수여기관 header
    covers three columns and the value sits in two of them, while the date that
    leaked into the third covers only one.
    """
    start, end = header
    width = len(grid[0]) if grid else 0
    groups = []
    for col in range(width):
        texts, span = [], (col, col)
        for row in grid[start : end + 1]:
            cell = row[col] if col < len(row) else None
            if cell is None or not cell.text:
                continue
            if cell.text not in texts:
                texts.append(cell.text)
            first, last = col, col
            while first > 0 and row[first - 1] is cell:
                first -= 1
            while last + 1 < len(row) and row[last + 1] is cell:
                last += 1
            span = (first, last)  # the deepest header row wins
        groups.append((" ".join(texts), span))
    return groups


def map_columns(groups, template) -> list[tuple[int, int] | None]:
    """Assign each template column the grid columns it should be read from.

    Left to right and never backwards, so a repeated header label (출결표's
    질병/미인정/기타 appear four times) cannot pull a later column to an
    earlier group.
    """
    assigned: list[tuple[int, int] | None] = []
    cursor = 0
    for column in template.columns:
        best, best_score = None, MIN_LABEL_SIMILARITY
        for index in range(cursor, len(groups)):
            text, span = groups[index]
            if not text:
                continue
            # Compare against the whole header stack ("결석일수 질병") and
            # against the deepest row alone ("일자 또는 기간"), whichever fits.
            score = max(similarity(text, column.label), similarity(text.split(" ")[-1], column.label))
            if score > best_score:
                best, best_score = span, score
        assigned.append(best)
        if best:
            cursor = best[1] + 1
    return assigned


def value_from_span(row, span: tuple[int, int]) -> str:
    """Read a header group's value out of a data row.

    Picks the non-empty cell covering the most columns of the group, so a value
    that merely spills into the group from its neighbour loses to the one that
    actually fills it.
    """
    first, last = span
    counts: dict[int, list] = {}
    for col in range(first, min(last + 1, len(row))):
        cell = row[col]
        if cell is None or not cell.text:
            continue
        entry = counts.setdefault(id(cell), [0, col, cell.text])
        entry[0] += 1
    if not counts:
        return ""
    return max(counts.values(), key=lambda e: (e[0], -e[1]))[2]


def score_values(template, values: list[str]) -> float:
    """Rate one candidate alignment of a row against the column kinds."""
    total = 0.0
    for column, value in zip(template.columns, values):
        value = (value or "").strip()
        if not value:
            continue
        validator, weight = KINDS[column.kind]
        total += weight if validator(value) else -1.0
    return total


def read_row(template, grid, row_index, spans) -> tuple[list[str], float]:
    """Pick the best alignment of one data row against the template's columns."""
    row = grid[row_index]
    candidates = []
    if spans:
        candidates.append([value_from_span(row, span) if span else "" for span in spans])
    # Fallback: take the row's cells in order. This is what saves the tables the
    # OCR padded with empty columns (출결표) or whose header it garbled.
    cells = logical_cells(row)
    width = len(template.columns)
    if cells:
        if len(cells) >= width:
            for offset in range(len(cells) - width + 1):
                candidates.append(cells[offset : offset + width])
        else:
            for offset in range(width - len(cells) + 1):
                candidates.append([""] * offset + cells + [""] * (width - len(cells) - offset))
    best, best_score = candidates[0], score_values(template, candidates[0])
    for candidate in candidates[1:]:
        score = score_values(template, candidate)
        if score > best_score:  # ties keep the earlier (header-aware) candidate
            best, best_score = candidate, score
    return best, best_score


def split_merged_counts(template, values: list[str]) -> list[str]:
    """Undo two adjacent numeric columns the OCR ran together ('시간 누계시간' -> '1 1')."""
    for index, column in enumerate(template.columns[:-1]):
        follower = template.columns[index + 1]
        if column.kind != "count" or follower.kind != "count" or values[index + 1].strip():
            continue
        m = re.fullmatch(r"\s*(\d{1,4})\s+(\d{1,4})\s*", values[index])
        if m:
            values[index], values[index + 1] = m.group(1), m.group(2)
    return values


def read_table(template, grid, start, spans, carry: dict) -> list[dict]:
    """Read every data row of a table into {column name: text} dicts."""
    rows = []
    for index in range(start, len(grid)):
        row = grid[index]
        previous = grid[index - 1] if index > 0 else []
        held = {id(cell) for cell in previous if cell is not None}
        # Cells this row introduces, as opposed to the ones a rowspan carried
        # into it from the row above.
        fresh = {c.text for c in row if c is not None and id(c) not in held and c.text}
        values, score = read_row(template, grid, index, spans)
        if score < MIN_ROW_SCORE:
            continue
        mapped = {}
        for column, value in zip(template.columns, split_merged_counts(template, values)):
            value = (value or "").strip()
            validator, _ = KINDS[column.kind]
            if value and (is_junk(value) or not validator(value)):
                value = ""  # 검증에 걸리면 잘못 밀려온 값이므로 버린다
            mapped[column.name] = value
        for column, span in zip(template.columns, spans or []):
            # A 학년/학기 cell held by a rowspan is explicit structure, so it
            # outranks whatever the alignment guessed for that column.
            cell = row[span[0]] if span and span[0] < len(row) else None
            if column.carry and cell is not None and id(cell) in held:
                if KINDS[column.kind][0](cell.text):
                    mapped[column.name] = cell.text
        for column in template.columns:
            if column.carry:
                mapped[column.name] = mapped[column.name] or carry.get(column.name, "")
                carry[column.name] = mapped[column.name]
        # A row whose key value was only inherited carries no new record: it is
        # the second half of a cell the OCR split (특기사항 서술만 있는 행).
        marked = any(similarity(text, label) >= 0.8 for text in fresh for label in template.keep_if)
        if not marked and not any(
            mapped[name] and mapped[name] in fresh for name in template.key_names
        ):
            continue
        mapped["_cells"] = logical_cells(row)
        mapped["_fresh"] = fresh
        rows.append(mapped)
    return rows


# ------------------------------------------------------- record construction


def split_outside_parens(text: str) -> list[str]:
    """Split on commas that are not inside brackets ('제목(저자, 저자), 제목(저자)')."""
    parts, depth, buffer = [], 0, ""
    for char in text:
        if char in "([{（〔":
            depth += 1
        elif char in ")]}）〕":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append(buffer)
            buffer = ""
        else:
            buffer += char
    parts.append(buffer)
    return [p.strip() for p in parts if p.strip()]


PROGRAM_RE = re.compile(r"^\(\s*([^()]{1,40}?)\s*\)")


def first_program(cells: list[str]) -> str | None:
    """The programme name a 특기사항 opens with: '(나의 꿈을 JOB아라2)(19시간) ...'.

    It is a value, not prose, and it is what tells two 자유학기 rows of the same
    영역 and 시간 apart (진로탐색A / 진로탐색B).
    """
    for cell in cells:
        m = PROGRAM_RE.match(cell)
        if m:
            return m.group(1)
    return None


PERSON_RE = {
    "name": re.compile(r"성\s*명\s*[:：]\s*(.+?)\s*(?=성\s*별|주민|주\s*소|$)"),
    "gender": re.compile(r"성\s*별\s*[:：]\s*(\S+?)\s*(?=주민|주\s*소|성\s*명|$)"),
    "resident_registration_number": re.compile(r"(\d{6}\s*-\s*\d{7})"),
    "address": re.compile(r"주\s*소\s*[:：]\s*(.+?)\s*$"),
}
HISTORY_RE = re.compile(r"(\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일)\s*(.+?)(?=\d{4}\s*년|$)")


class Extractor:
    """Walks the OCR blocks in order, filling the output JSON as it goes."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.data = {
            "student": {
                "name": None,
                "gender": None,
                "resident_registration_number": None,
                "address": None,
                "school_history": [],
                "enrollment": [],
            },
            "attendance": [],
            "academic_records": [],
            "non_subject_records": {
                "awards": [],
                "creative_activities": [],
                "volunteer_activities": [],
                "free_semester_activities": [],
                "reading_activities": [],
            },
        }
        self.section: str | None = None
        self.grade: int | None = None  # [1학년] 머리말 (교과학습발달상황용)
        self.carries: dict[str, dict] = {}
        self.last_match: tuple[Template, list] | None = None  # 페이지를 넘어 이어지는 표용
        self.stats: dict[str, int] = {}
        self.skipped = 0
        self.previous_table = None

    # -- context -------------------------------------------------------------

    def feed_text(self, block: Block) -> None:
        for line in block.text.splitlines():
            m = re.match(r"\s*(\d)\s*[.．]\s*(\S.*)$", line)
            if m and matches_label(m.group(2), SECTION_TITLES.get(m.group(1), [])):
                if self.section != m.group(1):
                    self.last_match = None
                self.section = m.group(1)
                self.log(f"  [p{block.page}] 절 인식: {m.group(1)}. {m.group(2)}")
            m = re.search(r"\[\s*(\d)\s*학\s*년\s*\]", line)
            if m:
                self.grade = int(m.group(1))
                self.log(f"  [p{block.page}] 학년 머리말: {self.grade}학년")

    # -- tables --------------------------------------------------------------

    def feed_table(self, block: Block) -> None:
        signature = re.sub(r"\s+", "", block.text)
        if signature == self.previous_table:
            return  # OCR이 같은 표를 연달아 토해낸 경우
        self.previous_table = signature
        grid = expand_table(block.text)
        if not grid or not any(distinct_texts(grid)):
            return
        if any(("성명" in t and ":" in t) or "학생정보" in t for t in distinct_texts(grid)):
            self.read_personal(grid, block.page)
            return

        best = None
        for template in TEMPLATES:
            header = find_header(grid, template)
            if not header:
                continue
            score = template_score(grid, header, template)
            # 절은 가려내는 조건이 아니라 우선순위다. 절 제목을 OCR이 놓쳐도
            # 머리글만 읽히면 표를 읽을 수 있어야 한다.
            score += 0.05 if template.section == self.section else 0.0
            if score >= MIN_TEMPLATE_SCORE and (best is None or score > best[0]):
                best = (score, template, header)

        if best:
            score, template, header = best
            spans = map_columns(column_groups(grid, header), template)
            start = header[1] + 1
        elif self.last_match and len(grid[0]) >= len(self.last_match[0].columns) - 1:
            # 머리글을 못 읽은 표(3학년 2학기 성적표처럼 글자가 깨진 경우)는
            # 같은 절에서 마지막으로 인식한 서식이 이어지는 것으로 보고 읽는다.
            # 열 수가 크게 줄었으면(세부능력 및 특기사항 표처럼) 다른 표다.
            template, spans, width = self.last_match
            score, start = 0.0, 0
            if width != len(grid[0]):
                spans = None  # 모양이 달라졌으면 열 대응은 버리고 위치 정렬로만 읽는다
        else:
            self.skipped += 1
            self.log(f"  [p{block.page}] 서식 미상 표 건너뜀 ({len(grid)}행)")
            return

        carry = self.carries.setdefault(template.name, {})
        rows = read_table(template, grid, start, spans, carry)
        if not rows:
            if best:
                self.log(f"  [p{block.page}] {template.name}: 데이터 행 없음")
            else:
                self.skipped += 1
            return
        for row in rows:
            self.emit(template, row)
        self.stats[template.name] = self.stats.get(template.name, 0) + len(rows)
        kind = "머리글" if best else "이어진 표"
        self.log(f"  [p{block.page}] {template.name} ({kind}, 일치도 {score:.2f}) {len(rows)}행")
        if best:
            self.last_match = (template, spans, len(grid[0]))

    def read_personal(self, grid, page: int) -> None:
        """1. 인적·학적사항: 머리글 없는 표라 셀 안의 라벨을 직접 찾는다."""
        student = self.data["student"]
        for text in distinct_texts(grid):
            if "성명" in text and ":" in text:
                for key, pattern in PERSON_RE.items():
                    m = pattern.search(text)
                    if m and not student[key]:
                        student[key] = m.group(1).strip()
            for date, description in HISTORY_RE.findall(text):
                description = description.strip()
                if not re.search(r"(입학|졸업|전입|전출|편입|검정)", description):
                    continue
                entry = {"date": re.sub(r"\s+", " ", date), "description": description}
                if entry not in student["school_history"]:
                    student["school_history"].append(entry)
        self.log(f"  [p{page}] 1. 인적·학적사항 읽음")

    # -- records -------------------------------------------------------------

    def emit(self, template, row: dict) -> None:
        cells = row.pop("_cells", [])
        fresh = row.pop("_fresh", set())
        name = template.name
        if name.startswith("학적"):
            self.append(
                self.data["student"]["enrollment"],
                {
                    "grade": to_int(row["grade"]),
                    "class_name": to_int(row["class_name"]),
                    "number": to_int(row["number"]),
                    "homeroom_teacher": blank(row["homeroom_teacher"]),
                },
            )
        elif name.startswith("2."):
            record = {"grade": to_int(row["grade"]), "school_days": to_int(row["school_days"])}
            for prefix in ("absence", "late", "early_leave", "result"):
                record[prefix] = {
                    key: to_int(row[f"{prefix}_{key}"]) for key in ("illness", "unexcused", "other")
                }
            record["remarks"] = blank(row["remarks"])
            self.append(self.data["attendance"], record)
        elif name.startswith("3."):
            self.append(
                self.data["non_subject_records"]["awards"],
                {
                    "grade": to_int(row["grade"]),
                    "title": blank(row["title"]),
                    "rank": blank(row["rank"]),
                    "date": blank(row["date"]),
                    "issuer": blank(row["issuer"]),
                    "participants": blank(row["participants"]),
                },
            )
        elif name.startswith("4. 창의적"):
            area, hours, desired = blank(row["area"]), to_int(row["hours"]), None
            # 진로활동 행은 특기사항 칸이 '희망분야 | 값' 으로 쪼개져 들어오고,
            # 앞 행의 rowspan이 영역·시간 자리를 차지하고 있을 때도 있다.
            for index, cell in enumerate(cells):
                if similarity(cell, "희망분야") >= 0.8:
                    desired = next((c for c in cells[index + 1 :] if c.strip()), None)
                    area = "진로활동"
                    numbers = [to_int(c) for c in cells[:index] if c in fresh and _count(c)]
                    hours = numbers[-1] if numbers else hours
                    break
            if not area:
                return
            self.append(
                self.data["non_subject_records"]["creative_activities"],
                {"grade": to_int(row["grade"]), "area": area, "hours": hours, "desired_field": desired},
            )
        elif name.startswith("4. 봉사"):
            self.append(
                self.data["non_subject_records"]["volunteer_activities"],
                {
                    "grade": to_int(row["grade"]),
                    "date_or_period": blank(row["date_or_period"]),
                    "place": blank(row["place"]),
                    "content": blank(row["content"]),
                    "hours": to_int(row["hours"]),
                    "cumulative_hours": to_int(row["cumulative_hours"]),
                },
            )
        elif name.startswith("5."):
            level = blank(row.get("level")) or ""
            raw, average = (row.get("score") or "").partition("/")[::2]
            self.append(
                self.data["academic_records"],
                {
                    "grade": self.grade,
                    "semester": to_int(row["semester"]),
                    "subject_group": blank(row["subject_group"]),
                    "subject": blank(row["subject"]),
                    "raw_score": to_float(raw),
                    "subject_average": to_float(average),
                    "achievement_level": (re.sub(r"\(.*", "", level).strip() or None),
                    "num_students": to_int(level.partition("(")[2]) if "(" in level else None,
                    "completed_hours": to_int(row.get("completed_hours")),
                    "remarks": blank(row.get("remarks")),
                    "area": name[name.find("(") + 1 : -1],
                },
                ("grade", "semester", "area", "subject"),
            )
        elif name.startswith("6."):
            self.append(
                self.data["non_subject_records"]["free_semester_activities"],
                {
                    "grade": to_int(row["grade"]),
                    "semester": to_int(row["semester"]),
                    "area": blank(row["area"]),
                    "program": first_program(cells),
                    "hours": to_int(row["hours"]),
                },
            )
        elif name.startswith("7."):
            books = row["books"]
            m = re.match(r"\s*\(\s*(\d)\s*학\s*기\s*\)\s*", books)
            self.append(
                self.data["non_subject_records"]["reading_activities"],
                {
                    "grade": to_int(row["grade"]),
                    "semester": int(m.group(1)) if m else None,
                    "subject_or_area": blank(row["subject_or_area"]),
                    "books": split_outside_parens(books[m.end() :] if m else books),
                },
            )

    @staticmethod
    def append(target: list, record: dict, key_fields: tuple = ()) -> None:
        """Add a record unless one already there matches it.

        The OCR repeats whole tables until it hits its token cap; those repeats
        carry no new information, and no row of this form is legitimately equal
        to another in every column. 교과 성적 matches on `key_fields` instead of
        the whole record, because a page that degenerates re-emits the same
        subject with a mangled 교과 name, which no equality test would catch.
        """
        if key_fields:
            key = tuple(record[name] for name in key_fields)
            if any(tuple(r[name] for name in key_fields) == key for r in target):
                return
        elif record in target:
            return
        target.append(record)

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr)


SECTION_TITLES = {
    "1": ["인적·학적사항", "인적사항", "인적학적사항"],
    "2": ["출결상황", "출 결 상황"],
    "3": ["수상경력"],
    "4": ["창의적 체험활동상황"],
    "5": ["교과 학습 발달 상황", "교과학습발달상황"],
    "6": ["자유학기활동상황"],
    "7": ["독서활동상황"],
    "8": ["행동특성 및 종합의견"],
}


def extract(text: str, verbose: bool = False) -> tuple[dict, Extractor]:
    extractor = Extractor(verbose)
    for block in iter_blocks(text):
        if block.is_table:
            extractor.feed_table(block)
        else:
            extractor.feed_text(block)
    return extractor.data, extractor


# ------------------------------------------------------------------------ CLI


def summarize(data: dict) -> str:
    non_subject = data["non_subject_records"]
    counts = {
        "학적(반/번호/담임)": len(data["student"]["enrollment"]),
        "학적사항": len(data["student"]["school_history"]),
        "출결(학년)": len(data["attendance"]),
        "교과 성적": len(data["academic_records"]),
        "수상경력": len(non_subject["awards"]),
        "창의적체험활동": len(non_subject["creative_activities"]),
        "봉사활동": len(non_subject["volunteer_activities"]),
        "자유학기활동": len(non_subject["free_semester_activities"]),
        "독서활동": len(non_subject["reading_activities"]),
    }
    return "\n".join(f"  {label}: {n}건" for label, n in counts.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="document_ocr.txt", help="OCR 원문 텍스트 파일")
    parser.add_argument("--output", default="document_extracted.json", help="결과 JSON 경로")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="어떤 표를 어느 서식으로 읽었는지 표별로 출력 (서식 점검용)",
    )
    args = parser.parse_args()

    try:
        with open(args.input, encoding="utf-8") as f:
            ocr_text = f.read()
    except OSError as e:
        sys.exit(f"error: 입력 파일을 읽을 수 없습니다: {e}")
    if not ocr_text.strip():
        sys.exit(f"error: 입력 파일이 비어 있습니다: {args.input}")

    data, extractor = extract(ocr_text, args.verbose)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    student = data["student"]
    print(f"저장: {args.output}")
    print(f"  학생: {student['name']} ({student['gender']})")
    print(summarize(data))
    if extractor.skipped:
        print(f"  (서식에 없는 표 {extractor.skipped}개는 건너뜀)", file=sys.stderr)


if __name__ == "__main__":
    main()

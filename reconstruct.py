"""Reconstruct a viewable HTML/PDF document from Unlimited-OCR output.

Reads the raw OCR text (with <|det|>type[bbox]<|/det|>content markers,
bbox normalized to 0-1000 relative to each page's own width/height) and
lays each block back onto an A4-sized page using absolute positioning,
so the parsed content (including HTML tables) can be visually compared
against the original PDF. Image-type blocks are filled with an actual
crop from the re-rendered original page.
"""

import argparse
import base64
import html
import io
import re
import subprocess
from pathlib import Path

import pymupdf as fitz
from PIL import Image

DET_RE = re.compile(r"<\|det\|>\s*([a-zA-Z_]+)\s*\[([^\]]+)\]<\|/det\|>")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

A4_PX_W = 793.7  # 210mm @ 96dpi
A4_PX_H = 1122.5  # 297mm @ 96dpi

TYPE_CLASS_STYLE = {
    "title": "font-weight:700;font-size:15px;",
    "header": "font-size:10px;color:#666;",
    "text": "font-size:11px;",
}


def render_pages(pdf_path: str, dpi: int) -> list[Image.Image]:
    zoom = dpi / 72
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            pages.append(img.convert("RGB"))
    return pages


def split_pages(raw_text: str) -> list[str]:
    parts = re.split(r"={5,}\s*Page\s+\d+\s*={5,}", raw_text)
    return [p.strip("\n") for p in parts if p.strip()]


def parse_blocks(page_text: str):
    matches = list(DET_RE.finditer(page_text))
    leading = page_text[: matches[0].start()].strip() if matches else page_text.strip()
    blocks = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        content = page_text[start:end].strip()
        coords = [float(c.strip()) for c in m.group(2).split(",")]
        if len(coords) != 4:
            continue
        blocks.append({"type": m.group(1), "bbox": coords, "content": content})
    return leading, blocks


def image_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def close_unclosed_table(content: str) -> str:
    """The model occasionally ends generation mid-table without emitting the
    closing tags (usually inside a very long free-text cell). Patch that up
    so one broken table doesn't swallow the rest of the HTML document."""
    opens = content.count("<table")
    closes = content.count("</table>")
    if opens <= closes:
        return content
    td_open = content.count("<td") - content.count("</td>")
    tr_open = content.count("<tr") - content.count("</tr>")
    patch = "</td>" * max(td_open, 0) + "</tr>" * max(tr_open, 0) + "</table>" * (opens - closes)
    return content + patch


def block_html(block, page_img: Image.Image) -> str:
    btype = block["type"]
    x1, y1, x2, y2 = block["bbox"]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    pos = (
        f"left:{x1 / 10:.3f}%;top:{y1 / 10:.3f}%;"
        f"width:{max(x2 - x1, 1) / 10:.3f}%;height:{max(y2 - y1, 1) / 10:.3f}%;"
    )

    if btype == "table":
        content = close_unclosed_table(block["content"])
        return f'<div class="block table-block" style="{pos}">{content}</div>'

    if btype == "image":
        W, H = page_img.size
        box = (
            max(0, int(x1 / 1000 * W)),
            max(0, int(y1 / 1000 * H)),
            min(W, int(x2 / 1000 * W)),
            min(H, int(y2 / 1000 * H)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return ""
        crop = page_img.crop(box)
        uri = image_to_data_uri(crop)
        return (
            f'<div class="block" style="{pos}">'
            f'<img src="{uri}" style="width:100%;height:100%;object-fit:contain;"></div>'
        )

    extra = TYPE_CLASS_STYLE.get(btype, "font-size:11px;")
    text = html.escape(block["content"]).replace("\n", "<br>")
    return f'<div class="block {html.escape(btype)}" style="{pos}{extra}">{text}</div>'


def page_html(index: int, leading: str, blocks: list, page_img: Image.Image) -> str:
    body = "\n".join(block_html(b, page_img) for b in blocks)
    leading_html = (
        f'<div class="leading">{html.escape(leading)}</div>' if leading else ""
    )
    return f'''<section class="page">
  <div class="page-label">Page {index}</div>
  {leading_html}
  {body}
</section>'''


CSS = f"""
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#ddd; font-family:"Apple SD Gothic Neo","Malgun Gothic",sans-serif; }}
.page {{
  position:relative; width:{A4_PX_W}px; height:{A4_PX_H}px;
  margin:24px auto; background:#fff; box-shadow:0 0 8px rgba(0,0,0,.35);
  page-break-after:always; overflow:hidden;
}}
.page-label {{ position:absolute; top:-18px; left:0; font-size:11px; color:#888; }}
.block {{ position:absolute; line-height:1.25; word-break:break-all; }}
.leading {{ position:absolute; top:2px; left:2px; font-size:8px; color:#ccc; max-width:98%; }}
.table-block table {{ border-collapse:collapse; width:100%; height:100%; font-size:9px; table-layout:fixed; }}
.table-block td, .table-block th {{ border:1px solid #999; padding:1px 3px; overflow:hidden; text-overflow:ellipsis; }}
@page {{ size: A4; margin:0; }}
@media print {{
  body {{ background:#fff; }}
  .page {{ margin:0; box-shadow:none; }}
  .page-label {{ display:none; }}
}}
"""


def build_html(pdf_path: str, ocr_text_path: str, dpi: int) -> str:
    raw_text = Path(ocr_text_path).read_text(encoding="utf-8")
    page_texts = split_pages(raw_text)
    page_imgs = render_pages(pdf_path, dpi)

    sections = []
    for i, page_text in enumerate(page_texts, start=1):
        leading, blocks = parse_blocks(page_text)
        page_img = page_imgs[i - 1] if i - 1 < len(page_imgs) else page_imgs[-1]
        sections.append(page_html(i, leading, blocks, page_img))

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>OCR Reconstruction</title>
<style>{CSS}</style></head>
<body>
{chr(10).join(sections)}
</body></html>"""


def html_to_pdf(html_path: Path, pdf_path: Path):
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",
            f"file://{html_path.resolve()}",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default="document.pdf")
    parser.add_argument("--ocr-text", default="document_ocr.txt")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--out-html", default="document_reconstructed.html")
    parser.add_argument("--out-pdf", default="document_reconstructed.pdf")
    parser.add_argument("--no-pdf", action="store_true", help="Skip Chrome PDF export")
    args = parser.parse_args()

    html_doc = build_html(args.pdf, args.ocr_text, args.dpi)
    out_html = Path(args.out_html)
    out_html.write_text(html_doc, encoding="utf-8")
    print(f"wrote {out_html}")

    if not args.no_pdf:
        out_pdf = Path(args.out_pdf)
        html_to_pdf(out_html, out_pdf)
        print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()

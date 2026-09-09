"""Parse a PDF with Unlimited-OCR (mlx-vlm) and print the extracted text.

Renders each page of the PDF to an image, runs the OCR model on it with
the "document parsing." task (which yields structured markdown, including
tables), and prints the result page by page.
"""

import argparse
import contextlib
import sys

import pymupdf as fitz
from PIL import Image, ImageDraw


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", default="document.pdf", help="Input PDF path")
    parser.add_argument("--model", default="mlx-community/Unlimited-OCR-8bit")
    parser.add_argument("--prompt", default="document parsing.")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
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
    args = parser.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    print(f"[loading model {args.model}]", file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        model, processor = load(args.model)
        config = load_config(args.model)
        formatted_prompt = apply_chat_template(processor, config, args.prompt, num_images=1)

    print(f"[rendering pages from {args.pdf} at {args.dpi} dpi]", file=sys.stderr)
    pages = render_pages(args.pdf, args.dpi, args.header_frac, args.footer_frac)
    print(f"[{len(pages)} page(s)]", file=sys.stderr)

    for i, page_image in enumerate(pages, start=1):
        print(f"[OCR page {i}/{len(pages)}]", file=sys.stderr)
        result = generate(
            model,
            processor,
            formatted_prompt,
            [page_image],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if len(pages) > 1:
            print(f"\n\n===== Page {i} =====\n")
        print(result.text)


if __name__ == "__main__":
    main()

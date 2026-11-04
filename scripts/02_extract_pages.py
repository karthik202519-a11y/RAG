"""Step 2: extract selected pages three ways and render them as images.

Usage:  python scripts/02_extract_pages.py <pdf_path> <out_dir> <page> [<page> ...]
Pages are 1-based PDF viewer numbers.

For each page we save:
  page_N.png          the page as a human sees it (our "ground truth")
  page_N_text.txt     plain text, in the order PyMuPDF reads it
  page_N_tables.md    tables PyMuPDF detected, as Markdown
"""
import sys
from pathlib import Path

import pymupdf


def main(pdf_path: str, out_dir: str, pages: list[int]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with pymupdf.open(pdf_path) as doc:
        for n in pages:
            page = doc[n - 1]

            # 2x zoom so small diagram labels stay readable
            page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(out / f"page_{n}.png")

            (out / f"page_{n}_text.txt").write_text(page.get_text("text"), encoding="utf-8")

            tables = page.find_tables().tables
            md = [f"## Table {i + 1} (bbox {tuple(round(v) for v in t.bbox)})\n\n{t.to_markdown()}"
                  for i, t in enumerate(tables)]
            (out / f"page_{n}_tables.md").write_text(
                "\n\n".join(md) if md else "(no tables detected)", encoding="utf-8")

            print(f"page {n}: {len(page.get_text().strip())} chars, {len(tables)} table(s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(p) for p in sys.argv[3:]])

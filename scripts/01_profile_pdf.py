"""Step 1: profile every page of a PDF so we know where to look closely.

Usage:  python scripts/01_profile_pdf.py <pdf_path> <output_csv>

We do NOT extract content here. We only measure each page.
"""
import sys
from pathlib import Path

import pymupdf
import pandas as pd


def profile_page(page: pymupdf.Page) -> dict:
    text = page.get_text("text")
    # Vector drawings: lines, rectangles, curves. Flowcharts and charts
    # are usually drawn this way, not stored as pictures.
    drawings = page.get_drawings()
    # Tables: PyMuPDF guesses table regions from ruling lines / text alignment.
    tables = page.find_tables()

    return {
        "page": page.number + 1,  # 1-based, matches what a PDF viewer shows
        "chars": len(text.strip()),
        "text_blocks": len(page.get_text("blocks")),
        "images": len(page.get_images(full=True)),
        "drawings": len(drawings),
        "tables": len(tables.tables),
    }


def main(pdf_path: str, out_csv: str) -> None:
    rows = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            rows.append(profile_page(page))

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Pages profiled: {len(df)}")
    print(df.describe().loc[["min", "50%", "max"]].round(0).to_string())
    print(f"\nPages with < 100 chars (possibly visual/scanned): {(df.chars < 100).sum()}")
    print(f"Pages with >= 1 detected table: {(df.tables > 0).sum()}")
    print(f"Pages with >= 50 drawing objects (diagram/chart candidates): {(df.drawings >= 50).sum()}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

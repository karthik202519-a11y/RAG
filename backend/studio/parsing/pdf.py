"""PDF -> structured pages (prose blocks + Markdown tables), with citation metadata.

Per page we produce a list of items (text blocks and tables) plus the printed page
number, the section title, and a `needs_vision` flag. Pages with many vector
drawings outside tables are likely diagrams or charts; their content cannot be
recovered from text, so they are rendered to PNG for a later vision step.
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

DRAWING_THRESHOLD = 50  # drawing objects outside tables => likely diagram/chart
MIN_PAGES_FOR_HEADER_DETECTION = 5
SECTION_RE = re.compile(r"^(\d+\.\d+\s{1,3}\S|Appendix [A-Z]\b)")
CTRL_RE = re.compile(r"[\x00-\x06\x08\x0b-\x1f\x7f-\x85\x87-\x9f]")


def clean(text: str) -> str:
    # \x07 is a tab-like separator in tables of contents, \x86 a list bullet.
    text = text.replace("\x07", " ").replace("\x86", "•")
    text = CTRL_RE.sub("", text)
    text = text.replace("\xad\n", "").replace("\xad", "")  # soft hyphen (+ line break)
    text = unicodedata.normalize("NFKC", text)  # fi/fl ligature glyphs -> "fi"/"fl"
    text = re.sub(r"(?m)^>\s+", "- ", text)  # bullet glyph came through as ">"
    return re.sub(r"[ \t]+", " ", text).strip()


def inside(rect: pymupdf.Rect, container: pymupdf.Rect) -> bool:
    return container.contains(rect) or container.intersects(rect) and (
        (rect & container).get_area() > 0.5 * rect.get_area())


def table_to_markdown(page: pymupdf.Page, table) -> str:
    """Rebuild the table from the text inside each cell's box.

    Merged cells come back as None from the detector; we leave them empty
    instead of repeating the merged text in every grid position.
    """
    grid = []
    for row in table.rows:
        cells = []
        for bbox in row.cells:
            if bbox is None:
                cells.append("")
            else:
                cells.append(clean(page.get_text("text", clip=pymupdf.Rect(bbox))).replace("\n", " "))
        grid.append(cells)
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    lines = []
    for i, r in enumerate(grid):
        r = r + [""] * (width - len(r))
        lines.append("| " + " | ".join(c.replace("|", "/") for c in r) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def running_lines(doc: pymupdf.Document) -> set[str]:
    """Header/footer text repeated on many pages (e.g. the handbook title).

    Needs several pages to mean anything: in a 1-3 page document every line
    appears on "most pages", so detection would delete the whole document.
    """
    if len(doc) < MIN_PAGES_FOR_HEADER_DETECTION:
        return set()
    counts: Counter = Counter()
    for page in doc:
        counts.update({clean(b[4]) for b in page.get_text("blocks") if b[6] == 0})
    return {t for t, n in counts.items() if n > 0.3 * len(doc) and not t.isdigit() and len(t) < 80}


def extract_page(doc_name: str, page: pymupdf.Page, running: set[str], img_dir: Path) -> dict:
    tables = page.find_tables().tables
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    # No sort=True: it sorts top-to-bottom and interleaves two-column pages.
    # The PDF's own content order was correct on every page we checked.
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    printed_page = None
    section = None
    items: list[dict] = []

    for b in blocks:
        rect = pymupdf.Rect(b[:4])
        raw = b[4]
        in_margin = rect.y0 < 60 or rect.y1 > page.rect.height - 60
        # The page number is sometimes its own block, sometimes the first line
        # of a block that also holds the footer title ("180\nNASA SYSTEMS...").
        first, _, rest = raw.partition("\n")
        if printed_page is None and in_margin and first.strip().isdigit():
            printed_page = first.strip()
            raw = rest
        text = clean(raw)
        if not text:
            continue
        if SECTION_RE.match(text) and rect.y0 < 60:
            section = text
            continue
        if text in running:
            continue
        if any(inside(rect, tr) for tr in table_rects):
            continue
        items.append({"type": "text", "text": text.replace("\n", " ")})

    for t in tables:
        md = table_to_markdown(page, t)
        if md:
            items.append({"type": "table", "bbox": [round(v) for v in t.bbox], "markdown": md})

    outside = sum(1 for d in page.get_drawings()
                  if not any(inside(pymupdf.Rect(d["rect"]), tr) for tr in table_rects))
    needs_vision = outside >= DRAWING_THRESHOLD
    image_path = None
    if needs_vision:
        img_dir.mkdir(parents=True, exist_ok=True)
        image_path = f"visual_pages/page_{page.number + 1}.png"  # relative to the document folder
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(str(img_dir / f"page_{page.number + 1}.png"))

    return {
        "doc": doc_name,
        "pdf_page": page.number + 1,
        "printed_page": printed_page,
        "section": section,
        "needs_vision": needs_vision,
        "drawings_outside_tables": outside,
        "image_path": image_path,
        "items": items,
    }


def extract_pdf(pdf_path: Path, doc_name: str, img_dir: Path, progress=None, cancelled=None) -> list[dict]:
    """Parse a PDF into pages. `progress(done, total)` is called per page and
    `cancelled()` is polled so a long extraction can be stopped between pages."""
    with pymupdf.open(pdf_path) as doc:
        running = running_lines(doc)
        pages = []
        for p in doc:
            if cancelled and cancelled():
                raise InterruptedError("cancelled")
            pages.append(extract_page(doc_name, p, running, img_dir))
            if progress:
                progress(len(pages), len(doc))
        return pages

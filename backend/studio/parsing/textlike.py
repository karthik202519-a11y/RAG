"""Parsers for non-PDF formats. Each returns pages in the same shape as the PDF extractor:

    {"doc", "pdf_page", "printed_page", "section", "needs_vision", "drawings_outside_tables",
     "image_path", "items": [{"type": "text", "text": ...} | {"type": "table", "markdown": ...}]}

Plain text and Markdown are one logical page; headings update `section`.
"""
import csv
import io
import json
import re
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _page(doc: str, n: int, items: list[dict], section: str | None = None) -> dict:
    return {"doc": doc, "pdf_page": n, "printed_page": None, "section": section, "needs_vision": False,
            "drawings_outside_tables": 0, "image_path": None, "items": items}


def _md_table(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    lines = []
    for i, r in enumerate(rows):
        r = [c.replace("|", "/").replace("\n", " ").strip() for c in r] + [""] * (width - len(r))
        lines.append("| " + " | ".join(r) + " |")
        if i == 0:
            lines.append("|" + "---|" * width)
    return "\n".join(lines)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_text(path: Path, doc: str, markdown: bool = False) -> list[dict]:
    text = _decode(path.read_bytes()).replace("\r\n", "\n")
    items: list[dict] = []
    section = None
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        first = block.split("\n", 1)[0]
        m = HEADING_RE.match(first) if markdown else None
        if m and section is None:
            section = m.group(2)
        items.append({"type": "text", "text": block})
    return [_page(doc, 1, items, section)]


def parse_docx(path: Path, doc: str) -> list[dict]:
    import docx  # python-docx

    d = docx.Document(str(path))
    items: list[dict] = []
    section = None
    for p in d.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if p.style is not None and p.style.name.lower().startswith("heading") and section is None:
            section = t
        items.append({"type": "text", "text": t})
    for table in d.tables:
        rows = [[c.text for c in r.cells] for r in table.rows]
        if rows:
            items.append({"type": "table", "bbox": None, "markdown": _md_table(rows)})
    return [_page(doc, 1, items, section)]


def parse_html(path: Path, doc: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(path.read_bytes()), "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer"]):
        tag.decompose()
    items: list[dict] = []
    section = None
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])] for tr in table.find_all("tr")]
        rows = [r for r in rows if r]
        table.replace_with(soup.new_string(""))
        if rows:
            items.append({"type": "table", "bbox": None, "markdown": _md_table(rows)})
    texts = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
        t = el.get_text(" ", strip=True)
        if not t:
            continue
        if el.name in ("h1", "h2", "h3", "h4") and section is None:
            section = t
        texts.append({"type": "text", "text": t})
    return [_page(doc, 1, texts + items, section)]


def parse_csv(path: Path, doc: str) -> list[dict]:
    text = _decode(path.read_bytes())
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    items = [{"type": "table", "bbox": None, "markdown": _md_table(rows)}] if rows else []
    return [_page(doc, 1, items)]


def parse_json(path: Path, doc: str) -> list[dict]:
    data = json.loads(_decode(path.read_bytes()))
    return [_page(doc, 1, [{"type": "text", "text": json.dumps(data, indent=2, ensure_ascii=False)}])]

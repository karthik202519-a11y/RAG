"""Parsing front door: choose a parser by file type, return normalized pages."""
from pathlib import Path

from . import textlike

# extension -> (parser id, needs a page-image folder)
SUPPORTED: dict[str, str] = {
    "pdf": "pymupdf", "txt": "plain_text", "md": "markdown", "markdown": "markdown",
    "docx": "python_docx", "html": "html_bs4", "htm": "html_bs4", "csv": "csv", "json": "json",
}


def supported_extensions() -> list[str]:
    return sorted(SUPPORTED)


def parser_for(ext: str) -> str | None:
    return SUPPORTED.get(ext.lower().lstrip("."))


def parse_file(path: Path, ext: str, doc_name: str, img_dir: Path | None = None,
               progress=None, cancelled=None) -> tuple[str, list[dict]]:
    """Return (parser id, pages)."""
    parser = parser_for(ext)
    if parser is None:
        raise ValueError(f"Unsupported file type: .{ext}")
    if parser == "pymupdf":
        from .pdf import extract_pdf
        return parser, extract_pdf(path, doc_name, img_dir or path.parent / "visual_pages", progress, cancelled)
    pages = {
        "plain_text": lambda: textlike.parse_text(path, doc_name),
        "markdown": lambda: textlike.parse_text(path, doc_name, markdown=True),
        "python_docx": lambda: textlike.parse_docx(path, doc_name),
        "html_bs4": lambda: textlike.parse_html(path, doc_name),
        "csv": lambda: textlike.parse_csv(path, doc_name),
        "json": lambda: textlike.parse_json(path, doc_name),
    }[parser]()
    if progress:
        progress(len(pages), len(pages))
    return parser, pages

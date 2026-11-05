"""Parsing front door: choose a parser by file type, return normalized pages."""
from pathlib import Path

from . import textlike

# extension -> (parser id, needs a page-image folder)
SUPPORTED: dict[str, str] = {
    "pdf": "pymupdf", "txt": "plain_text", "md": "markdown", "markdown": "markdown",
    "docx": "python_docx", "html": "html_bs4", "htm": "html_bs4", "csv": "csv", "json": "json",
    "pptx": "python_pptx", "xlsx": "openpyxl", "xml": "xml_etree",
    "png": "rapidocr", "jpg": "rapidocr", "jpeg": "rapidocr",
    **{ext: "source_code" for ext in textlike.CODE_EXTS},
}


def supported_extensions() -> list[str]:
    return sorted(SUPPORTED)


def parser_for(ext: str) -> str | None:
    return SUPPORTED.get(ext.lower().lstrip("."))


# Parsers an upload may choose instead of the default for its type.
ALTERNATIVES: dict[str, list[str]] = {
    "pdf": ["pymupdf", "pdfplumber", "rapidocr", "azure_document_intelligence"],
    "png": ["rapidocr"], "jpg": ["rapidocr"], "jpeg": ["rapidocr"],
}


def parser_options(ext: str) -> list[str]:
    return ALTERNATIVES.get(ext.lower().lstrip("."), [])


def parse_file(path: Path, ext: str, doc_name: str, img_dir: Path | None = None,
               progress=None, cancelled=None, parser: str | None = None) -> tuple[str, list[dict]]:
    """Return (parser id, pages). `parser` overrides the default for this file type when it is one of its alternatives."""
    default = parser_for(ext)
    if default is None:
        raise ValueError(f"Unsupported file type: .{ext}")
    if parser and parser != default:
        if parser not in parser_options(ext):
            raise ValueError(f"Parser '{parser}' cannot read .{ext} files. Choose one of: {', '.join(parser_options(ext)) or default}")
        from . import alt
        if parser == "pdfplumber":
            return parser, alt.extract_pdfplumber(path, doc_name, progress, cancelled)
        if parser == "rapidocr":
            return parser, (alt.extract_ocr_pdf(path, doc_name, progress, cancelled) if ext.lower() == "pdf" else alt.extract_ocr_image(path, doc_name))
        if parser == "azure_document_intelligence":
            return parser, alt.extract_azure_di(path, doc_name, progress, cancelled)
    parser = default
    if parser == "rapidocr":
        from . import alt
        pages = alt.extract_ocr_image(path, doc_name)
        if progress:
            progress(1, 1)
        return parser, pages
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
        "python_pptx": lambda: textlike.parse_pptx(path, doc_name),
        "openpyxl": lambda: textlike.parse_xlsx(path, doc_name),
        "xml_etree": lambda: textlike.parse_xml(path, doc_name),
        "source_code": lambda: textlike.parse_code(path, doc_name),
    }[parser]()
    if progress:
        progress(len(pages), len(pages))
    return parser, pages

"""Registry entries for document parsers (stage: extraction)."""
from .base import MethodSpec, ParamSpec, registry

C, STAGE = "parser", "Extraction"


def _p(id_, name, summary, how, adv, dis, ext, status="implemented", reason=None, creds=False, requires=None,
       params=None, docs=None):
    registry.register(MethodSpec(
        id=id_, category=C, stage=STAGE, name=name, summary=summary, how_it_works=how, advantages=adv,
        disadvantages=dis, suitable_documents=docs or [], inputs=ext, outputs=["pages of text blocks and tables"],
        status=status, unavailable_reason=reason, needs_cloud_credentials=creds, requires=requires or [],
        params=params or []))


_p("pymupdf", "PyMuPDF (native PDF text + tables)",
   "Reads the text already stored inside a digital PDF and rebuilds tables cell by cell.",
   "Extracts text blocks in the PDF's own reading order, detects tables, rebuilds each cell from the text inside its "
   "box, removes repeated headers and footers, captures printed page numbers and section titles, and flags pages "
   "with many vector drawings (diagrams, charts) for a later vision step.",
   ["Fast, local, no cost", "Keeps page numbers for citations"],
   ["Cannot read scanned pages (no OCR)", "Charts and diagrams contribute only their text labels",
    "Table detection depends on the PyMuPDF version"],
   ["pdf"], docs=["Digital PDFs"],
   requires=["pymupdf==2.0 with pymupdf-layout"])
_p("plain_text", "Plain text", "Reads .txt files; blank lines separate paragraphs.",
   "Decodes the bytes (UTF-8, UTF-16 or Windows-1252) and splits on blank lines.",
   ["Exact, trivial"], ["No structure beyond paragraphs"], ["txt"], docs=["Text files"])
_p("markdown", "Markdown", "Reads .md files, keeping paragraphs and noting the first heading as the section.",
   "Same as plain text, with Markdown headings recognised for the section title.",
   ["Preserves Markdown for header-based chunking"], ["Does not render Markdown"], ["md", "markdown"],
   docs=["Documentation"])
_p("python_docx", "DOCX (python-docx)", "Reads Word documents: paragraphs, headings and tables.",
   "Walks paragraphs and tables with python-docx; tables become Markdown tables.",
   ["Keeps headings and tables"], ["No page numbers (Word reflows pages)", "Headers/footers, text boxes and "
    "images are ignored"], ["docx"], docs=["Word documents"])
_p("html_bs4", "HTML (BeautifulSoup)", "Reads HTML pages: headings, paragraphs, lists and tables.",
   "Strips script, style and navigation, converts tables to Markdown and keeps text blocks in order.",
   ["Keeps tables"], ["No JavaScript rendering"], ["html", "htm"], docs=["Saved web pages"])
_p("csv", "CSV", "Reads a CSV file as one Markdown table.", "Parses rows with the csv module.",
   ["Exact"], ["Large CSVs become one huge table"], ["csv"], docs=["Tabular data"])
_p("json", "JSON", "Reads JSON as pretty-printed text.", "Parses and re-serializes with indentation.",
   ["Exact"], ["No record-level structure yet"], ["json"], docs=["Structured data"])

_NA = {
    "pdfplumber": ("pdfplumber", "Planned (Stage 5)."),
    "docling": ("Docling (layout-aware)", "Planned (Stage 5); needs the docling package and models."),
    "unstructured": ("Unstructured", "Planned (Stage 5)."),
    "tika": ("Apache Tika", "Planned (Stage 5); needs a Tika server."),
    "tesseract": ("Tesseract OCR", "Planned (Stage 5); scanned PDFs are not supported until OCR exists."),
    "paddleocr": ("PaddleOCR", "Planned (Stage 5)."),
    "azure_document_intelligence": ("Azure AI Document Intelligence", "Planned (Stage 5); needs Azure credentials."),
    "aws_textract": ("AWS Textract", "Planned (Stage 5); needs AWS credentials."),
    "google_document_ai": ("Google Document AI", "Planned (Stage 5); needs Google credentials."),
    "llamaparse": ("LlamaParse", "Planned (Stage 5); needs a LlamaCloud API key."),
    "python_pptx": ("PPTX", "Planned (Stage 2b); PowerPoint files are not yet accepted."),
    "xlsx": ("XLSX", "Planned (Stage 2b); Excel files are not yet accepted."),
}
for _id, (_name, _why) in _NA.items():
    _p(_id, _name, f"{_name} parser.", "Not implemented yet.", [], [], [], status="unavailable", reason=_why)

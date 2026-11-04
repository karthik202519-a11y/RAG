"""Canonical document text: all parsed items joined into one string with known offsets.

Chunkers cut this string and report character spans, so the UI can draw exact chunk
boundaries and measure overlap between neighbours.
"""
from dataclasses import dataclass, field

SEP = "\n\n"


@dataclass
class ItemSpan:
    start: int
    end: int
    page: int
    section: str | None
    type: str  # "text" | "table"


@dataclass
class Canon:
    text: str
    spans: list[ItemSpan]
    pages: list[dict]
    printed: dict[int, str | None] = field(default_factory=dict)

    def pages_for(self, start: int, end: int) -> list[int]:
        return sorted({s.page for s in self.spans if s.start < end and s.end > start})

    def section_at(self, start: int, end: int) -> str | None:
        for s in self.spans:
            if s.start < end and s.end > start:
                return s.section
        return None

    def kinds_in(self, start: int, end: int) -> set[str]:
        return {s.type for s in self.spans if s.start < end and s.end > start}

    def page_ranges(self) -> list[tuple[int, int, int]]:
        """(page, start, end) covering each page's items."""
        out: dict[int, list[int]] = {}
        for s in self.spans:
            r = out.setdefault(s.page, [s.start, s.end])
            r[0], r[1] = min(r[0], s.start), max(r[1], s.end)
        return [(p, r[0], r[1]) for p, r in sorted(out.items())]


def build_canon(pages: list[dict]) -> Canon:
    parts: list[str] = []
    spans: list[ItemSpan] = []
    pos = 0
    section = None
    for page in pages:
        section = page.get("section") or section
        for item in page["items"]:
            text = item["text"] if item["type"] == "text" else item["markdown"]
            if not text.strip():
                continue
            if parts:
                pos += len(SEP)
            spans.append(ItemSpan(pos, pos + len(text), page["pdf_page"], section, item["type"]))
            parts.append(text)
            pos += len(text)
    full = SEP.join(parts)
    assert len(full) == pos or not parts
    return Canon(full, spans, pages, {p["pdf_page"]: p.get("printed_page") for p in pages})

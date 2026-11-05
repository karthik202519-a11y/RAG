"""Canonical document text: all parsed items joined into one string with known offsets.

Chunkers cut this string and report character spans, so the UI can draw exact chunk
boundaries and measure overlap between neighbours.
"""
from bisect import bisect_right
from dataclasses import dataclass, field

SEP = "\n\n"


@dataclass
class ItemSpan:
    start: int
    end: int
    page: int
    section: str | None
    type: str  # "text" | "table"
    tag: str | None = None          # HTML element name for parsed web pages (h1, p, li ...)
    bbox: list | None = None        # [x0, y0, x1, y1] on the page, for PDFs parsed with layout metadata
    font: float | None = None       # dominant font size of the block, same condition


@dataclass
class Canon:
    text: str
    spans: list[ItemSpan]
    pages: list[dict]
    printed: dict[int, str | None] = field(default_factory=dict)
    _ends: list | None = field(default=None, repr=False)

    def _overlapping(self, start: int, end: int):
        """Spans overlapping [start, end). Spans are in document order and do not overlap, so binary search finds the
        first candidate; a linear scan here would make chunking quadratic on large documents."""
        if self._ends is None:
            self._ends = [s.end for s in self.spans]
        i = bisect_right(self._ends, start)
        while i < len(self.spans) and self.spans[i].start < end:
            yield self.spans[i]
            i += 1

    def pages_for(self, start: int, end: int) -> list[int]:
        return sorted({s.page for s in self._overlapping(start, end)})

    def section_at(self, start: int, end: int) -> str | None:
        for s in self._overlapping(start, end):
            return s.section
        return None

    def kinds_in(self, start: int, end: int) -> set[str]:
        return {s.type for s in self._overlapping(start, end)}

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
            spans.append(ItemSpan(pos, pos + len(text), page["pdf_page"], section, item["type"], item.get("tag"), item.get("bbox"), item.get("font")))
            parts.append(text)
            pos += len(text)
    full = SEP.join(parts)
    assert len(full) == pos or not parts
    return Canon(full, spans, pages, {p["pdf_page"]: p.get("printed_page") for p in pages})

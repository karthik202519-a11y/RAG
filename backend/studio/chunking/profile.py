"""Measure a document's structure and choose a chunking method from the measurements.

One place for the rules, used by the Adaptive chunker (to cut) and by the recommendation engine (to explain). Rules, not a model:
the same document always gets the same answer, and the answer carries the numbers that led to it.
"""
import re
import statistics
from collections import Counter

from .canon import Canon

HEAD_RE = re.compile(r"^(?:#{1,6}\s+\S.*|\d+(?:\.\d+){0,4}\.?\s+[A-Z][^\n]{2,90}|(?:Appendix|Chapter|Section|Part)\s+[A-Z0-9][^\n]{0,80}|[A-Z][A-Z0-9 ,:&/()'-]{5,80})$")
XML_LINE = re.compile(r"^[\w./-]+(\s\[[^\]]*\])?: ")


def is_heading(text: str) -> bool:
    t = text.strip()
    return "\n" not in t and len(t) <= 100 and not t.endswith((".", ",", ";")) and bool(HEAD_RE.match(t))


def json_records(text: str) -> bool:
    from .methods_extra import _json_records
    return text.lstrip()[:1] in ("{", "[") and _json_records(text) is not None


def profile(canon: Canon) -> dict:
    spans = canon.spans
    n = max(1, len(spans))
    chars = max(1, len(canon.text))
    tags = Counter(s.tag for s in spans if s.tag)
    boxed = [s for s in spans if s.bbox and s.font]
    body = statistics.median([s.font for s in boxed]) if boxed else None
    return {
        "blocks": len(spans), "chars": len(canon.text),
        "table_share": round(sum(s.end - s.start for s in spans if s.type == "table") / chars, 3),
        "tables": sum(1 for s in spans if s.type == "table"),
        "html_share": round(sum(v for k, v in tags.items() if k != "code") / n, 3),
        "code": tags.get("code", 0) > 0,
        "json_records": json_records(canon.text),
        "xml_records": (not json_records(canon.text)) and len(spans) >= 4
                       and sum(1 for s in spans[:30] if XML_LINE.match(canon.text[s.start:s.end])) >= min(len(spans), 30) * 0.8,
        "markdown_headings": len(re.findall(r"(?m)^#{1,6}\s+\S", canon.text)),
        "section_titles": len({s.section for s in spans if s.section}),
        "heading_like_lines": sum(1 for s in spans if s.type == "text" and s.end - s.start <= 100 and is_heading(canon.text[s.start:s.end])),
        "large_font_blocks": sum(1 for s in boxed if body and s.font >= body * 1.15 and s.end - s.start <= 120),
        "avg_block_chars": round(statistics.fmean(s.end - s.start for s in spans)) if spans else 0,
        "pages": len({s.page for s in spans}),
    }


def choose_method(p: dict) -> tuple[str, str]:
    """-> (method id, reason). The order matters: the most specific structure wins."""
    if p["code"]:
        return "code_aware", "the file is source code"
    if p["json_records"] or p["xml_records"]:
        return "json_xml_structure", "the file is a list of records (JSON or XML)"
    if p["html_share"] >= 0.5:
        return "html_structure", "the text came from an HTML page with headings, lists and tables"
    if p["markdown_headings"] >= 3:
        return "markdown_header", f"it has {p['markdown_headings']} Markdown headings"
    if p["table_share"] >= 0.2:
        return "structure_aware", f"tables make up {p['table_share']:.0%} of the text and must not be cut mid-row"
    if p["large_font_blocks"] >= 5:
        return "layout_aware", f"{p['large_font_blocks']} blocks are set in a larger font, so the page layout shows the headings"
    if p["section_titles"] >= 5 or p["heading_like_lines"] >= 8:
        return "section_based", "it has clear section structure"
    if p["avg_block_chars"] < 250 and p["blocks"] > 20:
        return "paragraph", "the text is many short blocks, so whole blocks are the natural unit"
    return "recursive", "no strong structure was detected, so the safe general-purpose method is used"


def describe(p: dict) -> str:
    return (f"{p['blocks']} blocks, {p['chars']:,} characters, tables {p['table_share']:.0%} of the text, {p['markdown_headings']} Markdown headings, "
            f"{p['heading_like_lines']} heading-like lines, {p['section_titles']} section titles, {p['large_font_blocks']} larger-font headings, "
            f"average block {p['avg_block_chars']} characters")

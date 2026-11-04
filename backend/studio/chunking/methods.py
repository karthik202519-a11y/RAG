"""Chunking method implementations.

Every method takes (canon, params) and returns (cuts, warnings). A cut is a dict
{"start", "end"} (character offsets into canon.text) and optionally {"text", "type"}
when the chunk text differs from the slice (e.g. a table with its caption prepended).
"""
import re
from collections.abc import Callable

from .canon import Canon
from .structure import build_chunks as structure_build
from .tokens import count, get_encoding

Cut = dict
SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
MD_HEADER_RE = re.compile(r"(?m)^(#{1,6})[ \t]+\S.*$")


# ----------------------------------------------------------------- helpers
def _measure(canon: Canon, unit: str, tokenizer: str) -> Callable[[int, int], int]:
    if unit == "characters":
        return lambda s, e: e - s
    return lambda s, e: count(canon.text[s:e], tokenizer)


def _sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    out, pos = [], start
    for m in SENT_RE.finditer(text, start, end):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < end:
        out.append((pos, end))
    return out


def _token_windows(canon: Canon, tokenizer: str, size: int, stride: int, a: int = 0, b: int | None = None
                   ) -> tuple[list[Cut], list[str]]:
    """Fixed windows of `size` tokens every `stride` tokens over canon.text[a:b]."""
    b = len(canon.text) if b is None else b
    enc = get_encoding(tokenizer)
    sub = canon.text[a:b]
    ids = enc.encode(sub, disallowed_special=())
    if not ids:
        return [], []
    decoded, offs = enc.decode_with_offsets(ids)
    warnings = []
    if decoded != sub:
        warnings.append("Tokenizer round-trip changed some characters; chunk offsets may be slightly off.")
    cuts, i = [], 0
    while i < len(ids):
        j = min(i + size, len(ids))
        start = a + offs[i]
        end = a + (offs[j] if j < len(ids) else len(sub))
        cuts.append({"start": start, "end": end})
        if j == len(ids):
            break
        i += stride
    return cuts, warnings


def _group_units(units: list[tuple[int, int]], size: int, overlap_units: int,
                 measure: Callable[[int, int], int]) -> list[Cut]:
    """Greedily group consecutive units up to `size`; repeat the last `overlap_units` units."""
    cuts: list[Cut] = []
    cur: list[tuple[int, int]] = []
    for u in units:
        if cur and measure(cur[0][0], u[1]) > size:
            cuts.append({"start": cur[0][0], "end": cur[-1][1]})
            keep = cur[-overlap_units:] if overlap_units else []
            while keep and measure(keep[0][0], u[1]) > size:
                keep = keep[1:]
            cur = keep
        cur.append(u)
    if cur and (not cuts or cur[-1][1] > cuts[-1]["end"]):
        cuts.append({"start": cur[0][0], "end": cur[-1][1]})
    return cuts


def _hard_split(canon: Canon, s: int, e: int, size: int, unit: str, tokenizer: str) -> list[tuple[int, int]]:
    if unit == "characters":
        return [(i, min(i + size, e)) for i in range(s, e, size)]
    cuts, _ = _token_windows(canon, tokenizer, size, size, s, e)
    return [(c["start"], c["end"]) for c in cuts]


# --------------------------------------------------------------- methods
def fixed_char(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    size, overlap, n = p["size"], p["overlap"], len(canon.text)
    cuts = [{"start": i, "end": min(i + size, n)} for i in range(0, n, size - overlap)
            if i == 0 or i + overlap < n]
    return cuts, []


def fixed_token(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    return _token_windows(canon, p["tokenizer"], p["size"], p["size"] - p["overlap"])


def sliding_window(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    return _token_windows(canon, p["tokenizer"], p["size"], p["stride"])


def recursive(canon: Canon, p: dict, a: int = 0, b: int | None = None) -> tuple[list[Cut], list[str]]:
    """Split canon.text[a:b] on paragraph breaks, then lines, sentences, words, then hard cuts."""
    b = len(canon.text) if b is None else b
    size, overlap, unit, tok = p["size"], p["overlap"], p["unit"], p["tokenizer"]
    measure = _measure(canon, unit, tok)
    text = canon.text
    seps = ["\n\n", "\n", ". ", " "]

    def split(s: int, e: int, level: int) -> list[tuple[int, int]]:
        if measure(s, e) <= size:
            return [(s, e)]
        if level >= len(seps):
            return _hard_split(canon, s, e, size, unit, tok)
        sep, out, pos = seps[level], [], s
        while True:
            i = text.find(sep, pos, e)
            piece_end = e if i < 0 else i + (1 if sep == ". " else 0)  # keep the full stop
            if piece_end > pos and text[pos:piece_end].strip():
                out.extend(split(pos, piece_end, level + 1))
            if i < 0:
                break
            pos = i + len(sep)
        return out

    pieces = split(a, b, 0)
    cuts: list[Cut] = []
    cur: list[tuple[int, int]] = []
    for pc in pieces:
        if cur and measure(cur[0][0], pc[1]) > size:
            cuts.append({"start": cur[0][0], "end": cur[-1][1]})
            keep: list[tuple[int, int]] = []
            for q in reversed(cur):
                if measure(q[0], cur[-1][1]) > overlap or measure(q[0], pc[1]) > size:
                    break
                keep.insert(0, q)
            cur = keep
        cur.append(pc)
    if cur and (not cuts or cur[-1][1] > cuts[-1]["end"]):
        cuts.append({"start": cur[0][0], "end": cur[-1][1]})
    return cuts, []


def sentence(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    measure = _measure(canon, "tokens", p["tokenizer"])
    units: list[tuple[int, int]] = []
    for sp in canon.spans:
        for s, e in _sentence_spans(canon.text, sp.start, sp.end):
            if measure(s, e) > p["size"]:
                units.extend(_hard_split(canon, s, e, p["size"], "tokens", p["tokenizer"]))
            else:
                units.append((s, e))
    warn = ["Sentence splitting is rule-based (. ! ?); abbreviations such as 'e.g.' can cause early splits."]
    return _group_units(units, p["size"], p["overlap_sentences"], measure), warn


def paragraph(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    measure = _measure(canon, "tokens", p["tokenizer"])
    units: list[tuple[int, int]] = []
    for sp in canon.spans:  # each parsed block is one paragraph
        if measure(sp.start, sp.end) <= p["size"]:
            units.append((sp.start, sp.end))
        else:
            for s, e in _sentence_spans(canon.text, sp.start, sp.end):
                if measure(s, e) > p["size"]:
                    units.extend(_hard_split(canon, s, e, p["size"], "tokens", p["tokenizer"]))
                else:
                    units.append((s, e))
    return _group_units(units, p["size"], p["overlap_paragraphs"], measure), []


def markdown_header(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    text = canon.text
    heads = [m for m in MD_HEADER_RE.finditer(text) if len(m.group(1)) <= p["max_level"]]
    if not heads:
        cuts, w = recursive(canon, {"size": p["size"], "overlap": p["overlap"], "unit": "tokens",
                                    "tokenizer": p["tokenizer"]})
        return cuts, ["No Markdown headers (# ...) found in this document, so the method fell back to "
                      "recursive splitting. PDF extraction does not produce # headers."] + w
    bounds = [0] + [m.start() for m in heads] + [len(text)]
    measure = _measure(canon, "tokens", p["tokenizer"])
    cuts: list[Cut] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if not text[s:e].strip():
            continue
        header_line = text[s:e].split("\n", 1)[0] if i > 0 else ""
        if measure(s, e) <= p["size"]:
            cuts.append({"start": s, "end": e})
            continue
        sub, _ = recursive(canon, {"size": p["size"], "overlap": p["overlap"], "unit": "tokens",
                                   "tokenizer": p["tokenizer"]}, s, e)
        for c in sub:
            cut = dict(c)
            if p["include_header"] and header_line and not text[c["start"]:c["end"]].startswith(header_line):
                cut["text"] = header_line + "\n" + text[c["start"]:c["end"]]
            cuts.append(cut)
    return cuts, []


def page_aware(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    ranges = canon.page_ranges()
    per, cuts = p["pages_per_chunk"], []
    for i in range(0, len(ranges), per):
        grp = ranges[i:i + per]
        s, e = grp[0][1], grp[-1][2]
        if p["max_tokens"] and count(canon.text[s:e], p["tokenizer"]) > p["max_tokens"]:
            measure = _measure(canon, "tokens", p["tokenizer"])
            units = [u for sp in canon.spans if sp.start >= s and sp.end <= e
                     for u in _sentence_spans(canon.text, sp.start, sp.end)]
            cuts.extend(_group_units(units, p["max_tokens"], 0, measure))
        else:
            cuts.append({"start": s, "end": e})
    return cuts, []


def structure_aware(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    chunks = structure_build(canon.pages, p["size"], p["overlap"])
    cuts: list[Cut] = []
    cursor, unlocated = 0, 0
    for c in chunks:
        lines = [ln for ln in c["text"].split("\n") if ln.strip() and not ln.startswith("Section: ")]
        start = end = None
        if c["type"] == "table":
            # The whole Markdown block is unique even when header rows repeat across pages of a continued table.
            md = "\n".join(ln for ln in c["text"].split("\n") if ln.startswith("|"))
            i = canon.text.find(md, max(0, cursor - 4000)) if md else -1
            if i < 0 and md:
                i = canon.text.find(md)
            if i >= 0:
                start, end, cursor = i, i + len(md), i
        # Otherwise locate by the first and last of the chunk's lines that exist verbatim in the document. Lines the
        # chunker added (a "continued from page N" caption, a section prefix) are skipped, not fatal.
        first = start is None and next(((canon.text.find(ln, max(0, cursor - 4000)), ln) for ln in lines
                      if canon.text.find(ln, max(0, cursor - 4000)) >= 0), None)
        if first:
            i = first[0]
            last = next(((canon.text.find(ln, i), ln) for ln in reversed(lines) if canon.text.find(ln, i) >= 0), None)
            if last:
                start, end = i, last[0] + len(last[1])
                cursor = start
        if start is None:
            unlocated += 1
        cuts.append({"start": start, "end": end, "text": c["text"], "type": c["type"],
                     "pages": c["pages"], "section": c.get("section")})
    warn = []
    if unlocated:
        warn.append(f"{unlocated} chunk(s) could not be located in the canonical text (shown without highlight).")
    return cuts, warn


METHODS: dict[str, Callable[[Canon, dict], tuple[list[Cut], list[str]]]] = {
    "fixed_char": fixed_char,
    "fixed_token": fixed_token,
    "sliding_window": sliding_window,
    "recursive": recursive,
    "sentence": sentence,
    "paragraph": paragraph,
    "markdown_header": markdown_header,
    "page_aware": page_aware,
    "structure_aware": structure_aware,
}

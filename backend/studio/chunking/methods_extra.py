"""The remaining chunking methods: overlap presets, HTML, section, topic, layout, code, JSON/XML, contextual,
proposition and adaptive.

Same contract as methods.py: (canon, params) -> (cuts, warnings), cuts are character spans into canon.text, optionally
with replacement text. Nothing here invents text: a chunk is a slice of the document, except where the method says so
(contextual adds a stored note, proposition rewrites and says it did).
"""
import ast
import json
import math
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .canon import Canon
from .methods import Cut, _hard_split, _measure, _sentence_spans, _token_windows, recursive
from .tokens import count

STOP = set("a an and are as at be by for from has have he her his how i in is it its of on or she that the their them there "
           "these they this to was were what when where which who why will with would you your not but can all any more".split())
WORD = re.compile(r"[a-z][a-z0-9_]{2,}")


# ------------------------------------------------------------------ shared packing
def _units_from_spans(canon: Canon, size: int, tok: str):
    """One unit per parsed block; a block longer than `size` is cut at sentences first so a unit never exceeds the budget."""
    measure = _measure(canon, "tokens", tok)
    out = []
    for sp in canon.spans:
        if sp.type == "table" or measure(sp.start, sp.end) <= size:
            out.append((sp.start, sp.end, sp))
        else:
            for s, e in _sentence_spans(canon.text, sp.start, sp.end):
                if measure(s, e) > size:
                    out.extend((a, b, sp) for a, b in _hard_split(canon, s, e, size, "tokens", tok))
                else:
                    out.append((s, e, sp))
    return out


def _pack(canon: Canon, units: list[tuple[int, int]], breaks: set[int], size: int, min_tokens: int, tok: str) -> list[Cut]:
    """Join consecutive units into chunks. A chunk ends before any unit index in `breaks` and when `size` would be exceeded;
    a chunk smaller than `min_tokens` is merged into its neighbour unless a hard break separates them."""
    measure = _measure(canon, "tokens", tok)
    groups: list[list[int]] = []
    for i, (s, e) in enumerate(units):
        if groups and i not in breaks and measure(units[groups[-1][0]][0], e) <= size:
            groups[-1].append(i)
        else:
            groups.append([i])
    merged: list[list[int]] = []
    for g in groups:
        small = measure(units[g[0]][0], units[g[-1]][1]) < min_tokens
        if merged and small and g[0] not in breaks and measure(units[merged[-1][0]][0], units[g[-1]][1]) <= size:
            merged[-1].extend(g)
        else:
            merged.append(g)
    return [{"start": units[g[0]][0], "end": units[g[-1]][1]} for g in merged]


# ------------------------------------------------------------------ 1-2. overlap presets
def overlapping(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    size = p["size"]
    ov = max(1, min(size - 1, round(size * p["overlap_percent"] / 100)))
    cuts, w = _token_windows(canon, p["tokenizer"], size, size - ov)
    return cuts, [f"Each chunk repeats the last {ov} tokens ({p['overlap_percent']}% of {size}) of the one before it."] + w


def non_overlapping(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    if p["strategy"] == "fixed_token":
        cuts, w = _token_windows(canon, p["tokenizer"], p["size"], p["size"])
    else:
        cuts, w = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]})
    return cuts, ["No text is repeated between chunks, so embedding cost equals the document's token count."] + w


# ------------------------------------------------------------------ 3. HTML structure
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def html_structure(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    tagged = [sp for sp in canon.spans if sp.tag]
    if not tagged:
        cuts, w = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]})
        return cuts, ["This document has no HTML element information (it was not parsed from an HTML file), so the method "
                      "fell back to recursive splitting."] + w
    units = _units_from_spans(canon, p["size"], p["tokenizer"])
    spans = [(s, e) for s, e, _ in units]
    breaks: set[int] = set()
    prev_kind = None
    for i, (s, e, sp) in enumerate(units):
        kind = "heading" if sp.tag in HEADING_TAGS else "table" if sp.type == "table" else "list" if sp.tag == "li" else "block"
        if kind == "heading" and p["split_on_headings"] or kind == "table" or prev_kind == "table":
            breaks.add(i)
        if kind != "list" and prev_kind == "list" and p["keep_lists_together"]:
            breaks.add(i)  # a list ends: close the chunk that holds it
        prev_kind = kind
    cuts = _pack(canon, spans, breaks, p["size"], p["min_tokens"], p["tokenizer"])
    heads = sum(1 for sp in tagged if sp.tag in HEADING_TAGS)
    return cuts, [f"Used {heads} heading(s), {sum(1 for sp in tagged if sp.tag == 'li')} list item(s) and "
                  f"{sum(1 for sp in canon.spans if sp.type == 'table')} table(s) as boundaries."]


# ------------------------------------------------------------------ 4. section based
HEAD_RE = re.compile(r"^(?:#{1,6}\s+\S.*|\d+(?:\.\d+){0,4}\.?\s+[A-Z][^\n]{2,90}|(?:Appendix|Chapter|Section|Part)\s+[A-Z0-9][^\n]{0,80}|[A-Z][A-Z0-9 ,:&/()'-]{5,80})$")


def _is_heading(text: str) -> bool:
    t = text.strip()
    return "\n" not in t and len(t) <= 100 and not t.endswith((".", ",", ";")) and bool(HEAD_RE.match(t))


def section_based(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    units = _units_from_spans(canon, p["size"], p["tokenizer"])
    breaks: set[int] = set()
    prev_sec = None
    heads = 0
    for i, (s, e, sp) in enumerate(units):
        is_head = sp.type == "text" and _is_heading(canon.text[s:e])
        if is_head or (sp.section != prev_sec and i > 0):
            breaks.add(i)
            heads += is_head
        prev_sec = sp.section
    cuts = _pack(canon, [(s, e) for s, e, _ in units], breaks, p["size"], p["min_tokens"], p["tokenizer"])
    secs = len({sp.section for sp in canon.spans if sp.section})
    warn = [f"Boundaries: {heads} heading-like line(s) and {secs} distinct section title(s) from the parser."]
    if not breaks:
        warn.append("No section structure was found, so the whole document was cut by size only.")
    return cuts, warn


# ------------------------------------------------------------------ 5. topic based (TextTiling)
def _terms(text: str) -> Counter:
    return Counter(w for w in WORD.findall(text.lower()) if w not in STOP)


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    return dot / (math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values())))


def topic_based(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    """TextTiling: compare the vocabulary of the k sentences before each gap with the k after it; a deep valley in that
    similarity is a change of topic. No model or embedding is used, so it is free and deterministic."""
    sents: list[tuple[int, int]] = []
    for sp in canon.spans:
        if sp.type == "table":
            sents.append((sp.start, sp.end))
        else:
            sents.extend(_sentence_spans(canon.text, sp.start, sp.end))
    sents = [u for u in sents if canon.text[u[0]:u[1]].strip()]
    k = p["block_sentences"]
    if len(sents) < 2 * k + 2:
        return [{"start": s, "end": e} for s, e in sents], ["Too little text to find topic changes; every sentence is its own chunk."]
    vec = [_terms(canon.text[s:e]) for s, e in sents]
    sims = []
    for i in range(1, len(sents)):
        left, right = Counter(), Counter()
        for v in vec[max(0, i - k):i]:
            left.update(v)
        for v in vec[i:i + k]:
            right.update(v)
        sims.append(_cos(left, right))
    depth = []
    for i, s in enumerate(sims):
        l = i
        while l > 0 and sims[l - 1] >= sims[l]:
            l -= 1
        r = i
        while r < len(sims) - 1 and sims[r + 1] >= sims[r]:
            r += 1
        depth.append((sims[l] - s) + (sims[r] - s))
    mu, sd = statistics.fmean(depth), statistics.pstdev(depth)
    cutoff = mu + p["sensitivity"] * sd
    breaks = {i + 1 for i, d in enumerate(depth) if d > cutoff and d > 0}
    breaks |= {i for i, (s, e) in enumerate(sents) if canon.kinds_in(s, e) == {"table"} or (i and canon.kinds_in(*sents[i - 1]) == {"table"})}
    cuts = _pack(canon, sents, breaks, p["max_tokens"], p["min_tokens"], p["tokenizer"])
    return cuts, [f"Found {len(breaks)} topic boundaries among {len(sents)} sentences (valley depth above mean {mu:.3f} + {p['sensitivity']} x std {sd:.3f}); "
                  f"then enforced {p['min_tokens']} to {p['max_tokens']} tokens."]


# ------------------------------------------------------------------ 6. layout aware
def layout_aware(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    spans = [sp for sp in canon.spans]
    boxed = [sp for sp in spans if sp.bbox]
    if len(boxed) < max(5, len(spans) // 4):
        cuts, w = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]})
        return cuts, ["This document has no page-layout metadata (positions and font sizes are recorded only for PDFs uploaded after "
                      "this feature, or the file is not a PDF), so the method fell back to recursive splitting. Re-upload the PDF to enable it."] + w
    fonts = [sp.font for sp in boxed if sp.font]
    body = statistics.median(fonts) if fonts else None
    heights = [sp.bbox[3] - sp.bbox[1] for sp in boxed if sp.bbox[3] > sp.bbox[1]]
    unit_h = statistics.median(heights) if heights else 12
    units, breaks = [], set()
    prev = None
    n_head = n_col = n_gap = 0
    for sp in spans:
        i = len(units)
        units.append((sp.start, sp.end))
        if prev is not None and sp.bbox and prev.bbox and sp.page == prev.page:
            if body and sp.font and sp.font >= body * p["heading_ratio"] and len(canon.text[sp.start:sp.end]) <= 120:
                breaks.add(i); n_head += 1
            elif sp.bbox[0] > prev.bbox[2] - 5 and sp.bbox[1] < prev.bbox[3]:
                breaks.add(i); n_col += 1  # the next block starts to the right of and above the previous one: a new column
            elif sp.bbox[1] - prev.bbox[3] > p["gap_factor"] * unit_h:
                breaks.add(i); n_gap += 1
        elif prev is not None and sp.page != prev.page and p["split_pages"]:
            breaks.add(i)
        if sp.type == "table" or (prev is not None and prev.type == "table"):
            breaks.add(i)
        prev = sp
    # oversized blocks are cut at sentences so every unit fits
    measure = _measure(canon, "tokens", p["tokenizer"])
    fine, fine_breaks = [], set()
    for i, (s, e) in enumerate(units):
        if i in breaks:
            fine_breaks.add(len(fine))
        if measure(s, e) > p["size"] and spans[i].type != "table":
            fine.extend(_sentence_spans(canon.text, s, e))
        else:
            fine.append((s, e))
    cuts = _pack(canon, fine, fine_breaks, p["size"], p["min_tokens"], p["tokenizer"])
    return cuts, [f"Layout boundaries: {n_head} heading(s) (font >= {p['heading_ratio']} x body {body}), {n_col} column change(s), "
                  f"{n_gap} large vertical gap(s), plus tables."]


# ------------------------------------------------------------------ 7. code aware
DECL_RE = re.compile(r"^(?:export\s+)?(?:default\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|static\s+|final\s+|abstract\s+|async\s+|pub\s+|unsafe\s+|extern\s+)*"
                     r"(?:def|class|function|func|fn|impl|struct|enum|interface|trait|type|module|namespace|object|fun|const|let|var|package|"
                     r"void|int|char|long|double|float|bool|string|String|auto|template|public|private)\b")
LEAD_RE = re.compile(r"^\s*(?://|/\*|\*|#|@|--)")


def _python_units(code: str, base: int) -> list[tuple[int, int]] | None:
    """Top-level statements from the syntax tree, with decorators and comment lines attached to what follows."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.split("\n")
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)
    units: list[tuple[int, int]] = []
    pending: int | None = None  # first line of a run of plain statements (imports, assignments)
    last_end = 0
    for node in tree.body:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        while first > last_end and LEAD_RE.match(lines[first - 1]):
            first -= 1
        is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if is_def:
            if pending is not None:
                units.append((pending, first)); pending = None
            units.append((first, node.end_lineno))
        elif pending is None:
            pending = first
        last_end = node.end_lineno
    if pending is not None:
        units.append((pending, last_end))
    return [(base + offs[a], base + min(offs[b], len(code))) for a, b in units if offs[b] > offs[a]]


def _brace_units(code: str, base: int) -> list[tuple[int, int]]:
    lines = code.split("\n")
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln) + 1)
    starts = [0]
    prev_blank = True
    for i, ln in enumerate(lines):
        top = ln and not ln[0].isspace() and not ln.startswith(("}", ")", "]", "#endif", "#else"))
        if top and i > 0 and prev_blank_or_close(lines, i) and (DECL_RE.match(ln) or LEAD_RE.match(ln)):
            j = i
            while j - 1 > starts[-1] and LEAD_RE.match(lines[j - 1]):  # keep the comment block with its declaration
                j -= 1
            if j > starts[-1]:
                starts.append(j)
    starts.append(len(lines))
    return [(base + offs[a], base + min(offs[b], len(code))) for a, b in zip(starts, starts[1:]) if offs[b] > offs[a]]


def prev_blank_or_close(lines: list[str], i: int) -> bool:
    j = i - 1
    if j >= 0 and not lines[j].strip():
        return True
    return j >= 0 and lines[j].startswith(("}", ")")) or (j >= 0 and LEAD_RE.match(lines[j]) is not None and prev_blank_or_close(lines, j))


def code_aware(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    code_spans = [sp for sp in canon.spans if sp.tag == "code"] or ([canon.spans[0]] if len(canon.spans) == 1 else [])
    if not code_spans:
        cuts, w = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]})
        return cuts, ["This document was not parsed as source code (upload a .py, .js, .ts, .java, .go, .rs, .c ... file), so the method "
                      "fell back to recursive splitting."] + w
    measure = _measure(canon, "tokens", p["tokenizer"])
    cuts: list[Cut] = []
    n_def = 0
    method = "syntax tree" if p["language"] in ("auto", "python") else "declaration patterns"
    for sp in code_spans:
        code = canon.text[sp.start:sp.end]
        units = _python_units(code, sp.start) if p["language"] in ("auto", "python") else None
        if units is None:
            units = _brace_units(code, sp.start)
            method = "declaration patterns" if p["language"] != "python" else method
        n_def += len(units)
        fine: list[tuple[int, int]] = []
        for s, e in units:
            if measure(s, e) <= p["size"]:
                fine.append((s, e))
            else:  # a function longer than the budget: cut at line breaks, keeping its first line as context
                sub, _ = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]}, s, e)
                head = canon.text[s:e].split("\n", 1)[0]
                for c in sub:
                    cut = {"start": c["start"], "end": c["end"]}
                    if p["repeat_signature"] and c["start"] > s:
                        cut["text"] = head + "\n    ...\n" + canon.text[c["start"]:c["end"]]
                    cuts.append(cut)
        cuts.extend(_pack(canon, fine, set(), p["size"] if p["merge_small"] else 0, p["min_tokens"] if p["merge_small"] else 0, p["tokenizer"])
                    if p["merge_small"] else [{"start": s, "end": e} for s, e in fine])
    cuts.sort(key=lambda c: c["start"])
    return cuts, [f"Found {n_def} top-level definition(s)/blocks using {method}; small neighbours were merged up to {p['size']} tokens."
                  if p["merge_small"] else f"Found {n_def} top-level definition(s)/blocks using {method}; one chunk each."]


# ------------------------------------------------------------------ 8. JSON / XML structure
def _json_records(text: str) -> list[tuple[int, int]] | None:
    dec = json.JSONDecoder()
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, (list, dict)):
        return None

    def scan(pos: int) -> list[tuple[int, int]]:
        """Spans of the direct children of the container that starts at text[pos]."""
        opener = text[pos]
        out, i = [], pos + 1
        while True:
            while i < len(text) and text[i] in " \t\r\n,":
                i += 1
            if i >= len(text) or text[i] in "]}":
                return out
            start = i
            if opener == "{":
                _, i = dec.raw_decode(text, i)       # the key
                while text[i] in " \t\r\n:":
                    i += 1
            _, i = dec.raw_decode(text, i)
            out.append((start, i))

    top = text.find("[") if isinstance(data, list) else text.find("{")
    recs = scan(top)
    # {"items": [ ... ]}: one key that holds the real list of records
    if isinstance(data, dict) and len(recs) == 1 and isinstance(next(iter(data.values())), list):
        s, e = recs[0]
        inner = text.find("[", s)
        recs = scan(inner)
    return recs


def _xml_line_records(canon: Canon) -> list[tuple[int, int]] | None:
    lines = []
    pos = 0
    for ln in canon.text.split("\n"):
        if ln.strip():  # parsed blocks are separated by blank lines
            lines.append((pos, pos + len(ln), ln.split(":", 1)[0].split(" [", 1)[0]))
        pos += len(ln) + 1
    if len(lines) < 4 or not all("/" in l[2] or l[2].isidentifier() for l in lines[:20]):
        return None
    depth_paths = Counter(l[2] for l in lines)
    start_path = lines[0][2]
    if depth_paths[start_path] < 2:
        # the first path is the container's own text; use the most common first-in-record path after it
        start_path = max(depth_paths, key=lambda k: depth_paths[k] if k.count("/") >= 2 else 0)
        if depth_paths[start_path] < 2:
            return None
    recs, cur_start, seen = [], None, set()
    for s, e, path in lines:
        if path == start_path and (cur_start is None or path in seen):
            if cur_start is not None:
                recs.append((cur_start, last_end))
            cur_start, seen = s, set()
        seen.add(path)
        last_end = e
    if cur_start is not None:
        recs.append((cur_start, last_end))
    return recs if len(recs) >= 2 else None


def json_xml_structure(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    text = canon.text
    recs = _json_records(text) if text.lstrip()[:1] in ("{", "[") else _xml_line_records(canon)
    if not recs or len(recs) < 2:
        cuts, w = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]})
        return cuts, ["No repeating records (a JSON array/object of items, or repeated XML elements) were found, so the method fell back to recursive splitting."] + w
    measure = _measure(canon, "tokens", p["tokenizer"])
    fine = []
    for s, e in recs:
        if measure(s, e) <= p["size"]:
            fine.append((s, e))
        else:
            sub, _ = recursive(canon, {"size": p["size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]}, s, e)
            fine.extend((c["start"], c["end"]) for c in sub)
    if p["records_per_chunk"] > 1:
        cuts = [{"start": fine[i][0], "end": fine[min(i + p["records_per_chunk"], len(fine)) - 1][1]} for i in range(0, len(fine), p["records_per_chunk"])]
    else:
        cuts = _pack(canon, fine, set(), p["size"], 0, p["tokenizer"]) if p["pack"] else [{"start": s, "end": e} for s, e in fine]
    return cuts, [f"Found {len(recs)} records; each stays whole unless it exceeds {p['size']} tokens."]


# ------------------------------------------------------------------ 9. contextual
def contextual(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    """Cut like Recursive, then ask the chat model for one situating sentence per chunk. The chunk text is unchanged; the sentence is
    stored beside it and is what gets embedded together with the chunk when an index is built from this method."""
    from ..config import settings
    from ..generation import service as gen
    from .contextual import situate

    ok, why = gen.generation_status()
    if not ok:
        raise ValueError(f"Contextual chunking needs the chat model. {why}")
    base, _ = recursive(canon, {"size": p["size"], "overlap": p["overlap"], "unit": "tokens", "tokenizer": p["tokenizer"]})
    model = p["model"] or settings.azure_chat_deployment
    limit = len(base) if p["max_chunks"] == 0 else min(len(base), p["max_chunks"])
    section = lambda c: canon.section_at(c["start"], c["end"])  # noqa: E731
    name = canon.pages[0].get("doc", "document") if canon.pages else "document"

    def one(i: int):
        c = base[i]
        before = canon.text[max(0, c["start"] - 500):c["start"]]
        after = canon.text[c["end"]:c["end"] + 500]
        try:
            return i, situate(name, section(c), before, canon.text[c["start"]:c["end"]], after, model)[0]
        except Exception:
            return i, None

    notes: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(settings.eval_workers, limit or 1))) as pool:
        for i, note in pool.map(one, range(limit)):
            if note:
                notes[i] = note
    cuts = [dict(c, context=notes.get(i)) for i, c in enumerate(base)]
    warn = [f"Wrote context for {len(notes)} of {len(base)} chunks with {model}."]
    if limit < len(base):
        warn.append(f"Only the first {limit} chunks were enriched (max chunks = {p['max_chunks']}); set it to 0 to enrich all {len(base)} (one chat call each).")
    if len(notes) < limit:
        warn.append(f"{limit - len(notes)} chunk(s) got no context because the model call failed.")
    return cuts, warn


# ------------------------------------------------------------------ 10. proposition
PROP_SYSTEM = ("Rewrite the passage as a list of short, self-contained factual statements (propositions). Each must make sense on its "
               "own: replace pronouns and vague references with the full names used in the passage. Use only facts stated in the passage. "
               'Reply with JSON only: {"propositions": ["...", "..."]}.')


def _content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", s.lower()) if w not in STOP}


def proposition(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    from ..config import settings
    from ..generation import service as gen

    ok, why = gen.generation_status()
    if not ok:
        raise ValueError(f"Proposition chunking needs the chat model. {why}")
    base, _ = recursive(canon, {"size": p["source_size"], "overlap": 0, "unit": "tokens", "tokenizer": p["tokenizer"]})
    limit = len(base) if p["max_passages"] == 0 else min(len(base), p["max_passages"])
    model = p["model"] or settings.azure_chat_deployment

    def one(i: int):
        c = base[i]
        src = canon.text[c["start"]:c["end"]]
        try:
            res = gen.chat_complete(model, [{"role": "system", "content": PROP_SYSTEM}, {"role": "user", "content": src}], 0.0, 900)
            m = re.search(r"\{.*\}", res.text, re.S)
            props = json.loads(m.group(0)).get("propositions", []) if m else []
        except Exception:
            return i, None, 0
        words = _content_words(src)
        good, dropped = [], 0
        for t in props:
            t = str(t).strip()
            cw = _content_words(t)
            if t and cw and len(cw & words) / len(cw) >= p["min_overlap"]:
                good.append(t)
            else:
                dropped += 1
        return i, good, dropped

    cuts: list[Cut] = []
    failed = dropped_total = 0
    with ThreadPoolExecutor(max_workers=max(1, min(settings.eval_workers, limit or 1))) as pool:
        for i, good, dropped in pool.map(one, range(limit)):
            if good is None:
                failed += 1
                continue
            dropped_total += dropped
            c = base[i]
            pages, sec = canon.pages_for(c["start"], c["end"]), canon.section_at(c["start"], c["end"])
            for j in range(0, len(good), p["per_chunk"]):
                cuts.append({"start": None, "end": None, "text": " ".join(good[j:j + p["per_chunk"]]), "type": "text",
                             "pages": pages, "section": sec})
    warn = [f"Rewrote {limit - failed} of {len(base)} passages into {len(cuts)} chunk(s) with {model}. Chunk text is the model's wording, so chunks are not "
            f"highlighted in the document; each statement was kept only if at least {int(p['min_overlap'] * 100)}% of its content words occur in its source passage "
            f"({dropped_total} dropped)."]
    if limit < len(base):
        warn.append(f"Only the first {limit} passages were processed (max passages = {p['max_passages']}); set it to 0 for all {len(base)} (one chat call each).")
    if failed:
        warn.append(f"{failed} passage(s) failed.")
    return cuts, warn


# ------------------------------------------------------------------ 11. adaptive
def adaptive(canon: Canon, p: dict) -> tuple[list[Cut], list[str]]:
    """Look at the document, pick the method that suits its structure, run it, and say why. Rules, not a model."""
    from ..registry import registry
    from . import engine
    from .methods import METHODS
    from .profile import choose_method, describe, profile

    prof = profile(canon)
    choice, why = choose_method(prof)
    spec = registry.get(choice)
    params = {"size": p["size"]} if any(sp.name == "size" for sp in spec.params) else {}
    cp = engine.validate_params(choice, params)
    cuts, w = METHODS[choice](canon, cp)
    return cuts, [f"Chose '{choice}' because {why}.", f"Measured: {describe(prof)}."] + w

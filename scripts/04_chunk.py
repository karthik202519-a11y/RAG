"""Step 5: turn extracted pages into chunks. Chunk size is a parameter.

Usage:  python scripts/04_chunk.py <extracted_json> <out_dir> <target_tokens> [overlap_tokens]

Rules
  - Prose blocks are grouped (in reading order, across pages) up to target_tokens.
  - A table is never merged with prose. It stays whole; if it exceeds the
    table limit it is split by rows and the header rows are repeated.
  - On diagram pages (needs_vision) the loose labels ("?", "4.1") are dropped;
    captions and sentences are kept. Diagram *content* comes from the vision step later.
  - Every chunk records its pages, section, and type so answers can be cited.
"""
import json
import re
import sys
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")  # tokenizer of text-embedding-3-small
TABLE_MAX_TOKENS = 1200
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


TABLE_CAP_RE = re.compile(r"^(TABLE|Table)\s+[\dA-Z][\w.‐‑‒–\-]*")
TABLE_NOTE_RE = re.compile(r"^(Note|Notes|Source|Sources|\*|†)\b|^\*")
CTRL_RE = re.compile(r"[\x00-\x06\x08\x0b-\x1f\x7f-\x85\x87-\x9f]")
MIN_CHUNK_TOKENS = 50


def ntok(text: str) -> int:
    return len(ENC.encode(text))


def scrub(text: str) -> str:
    """\\x07 is a tab-like separator in the table of contents; \\x86 is a list bullet."""
    return CTRL_RE.sub("", text.replace("\x07", " ").replace("\x86", "•"))


def pick_captions(items: list[dict]) -> dict[int, str]:
    """Map table item index -> caption text ("TABLE 2.1-1 ...") from the same page.

    Captions are short lines starting with TABLE; a sentence that merely mentions
    a table is longer. With as many captions as tables they pair in order,
    otherwise each table takes the closest caption above it (else below).
    """
    tabs = [k for k, i in enumerate(items) if i["type"] == "table"]
    caps = [k for k, i in enumerate(items)
            if i["type"] == "text" and TABLE_CAP_RE.match(i["text"]) and len(i["text"]) < 160]
    out = {}
    if len(caps) == len(tabs):
        out = {t: items[c]["text"] for t, c in zip(tabs, caps)}
    else:
        for t in tabs:
            above = [c for c in caps if c < t]
            below = [c for c in caps if c > t]
            c = max(above) if above else (min(below) if below else None)
            if c is not None:
                out[t] = items[c]["text"]
    return out


def table_notes(items: list[dict], k: int) -> str:
    """Footnote-like text directly after the table (Note:/Source:/*)."""
    notes = []
    for j in range(k + 1, min(len(items), k + 4)):
        it = items[j]
        if it["type"] == "text" and len(it["text"]) < 500 and TABLE_NOTE_RE.match(it["text"]):
            notes.append(it["text"])
        else:
            break
    return "\n".join(notes)


def is_label_noise(text: str) -> bool:
    """On a diagram page, drop only tiny loose labels such as "?", "4.1", "6.8".

    A first attempt that also dropped short non-sentence lines removed real
    content (list items, footnotes, "…can include the following:"), so we keep
    everything longer than this and accept some label text in the chunks.
    """
    return len(text) <= 5


def split_long(text: str, limit: int) -> list[str]:
    """Split a block longer than `limit` tokens on sentence boundaries."""
    if ntok(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sent in SENTENCE_RE.split(text):
        if cur and ntok(cur + " " + sent) > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    return parts


def split_table(markdown: str, limit: int) -> list[str]:
    if ntok(markdown) <= limit:
        return [markdown]
    lines = markdown.split("\n")
    header, rows = lines[:2], lines[2:]  # header row + |---| separator
    parts, cur = [], []
    for row in rows:
        if cur and ntok("\n".join(header + cur + [row])) > limit:
            parts.append("\n".join(header + cur))
            cur = []
        cur.append(row)
    if cur:
        parts.append("\n".join(header + cur))
    return parts


def merge_tiny(chunks: list[dict], max_tokens: int) -> list[dict]:
    """Fold text chunks under MIN_CHUNK_TOKENS into a neighbouring text chunk.

    Prefers the following chunk (a heading belongs to what follows it), else the
    previous one. Lines already shared through overlap are not duplicated.
    """
    out: list[dict] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if c["type"] == "text" and ntok(c["text"]) < MIN_CHUNK_TOKENS:
            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            if nxt and nxt["type"] == "text" and ntok(c["text"] + nxt["text"]) <= max_tokens:
                have = set(nxt["text"].split("\n"))
                lines = [l for l in c["text"].split("\n") if l not in have]
                nxt["text"] = "\n".join(lines + [nxt["text"]])
                nxt["pages"] = sorted(set(c["pages"]) | set(nxt["pages"]))
                i += 1
                continue
            prev = out[-1] if out else None
            if prev and prev["type"] == "text" and ntok(prev["text"] + c["text"]) <= max_tokens:
                have = set(prev["text"].split("\n"))
                prev["text"] += "\n" + "\n".join(l for l in c["text"].split("\n") if l not in have)
                prev["pages"] = sorted(set(c["pages"]) | set(prev["pages"]))
                i += 1
                continue
        out.append(c)
        i += 1
    for n, c in enumerate(out):
        c["chunk_id"] = n
    return out


def build_chunks(pages: list[dict], target: int, overlap: int) -> list[dict]:
    chunks = []
    buf: list[dict] = []  # blocks waiting to be flushed
    section = None

    def emit(kind, text, blocks_or_pages, section_):
        pages_ = sorted({b["page"] for b in blocks_or_pages}) if kind == "text" else blocks_or_pages
        chunks.append({
            "chunk_id": len(chunks),
            "type": kind,
            "section": section_,
            "pages": pages_,
            "text": text,
        })

    def flush(keep_overlap: bool):
        nonlocal buf
        if not buf:
            return
        emit("text", "\n".join(b["text"] for b in buf), buf, buf[0]["section"])
        if keep_overlap and overlap:
            tail, used = [], 0
            for b in reversed(buf):
                if used + b["tokens"] > overlap:
                    break
                tail.insert(0, b)
                used += b["tokens"]
            buf = tail
        else:
            buf = []

    last_table = None  # (page, first line, caption) of the previous table, for "continued" links
    for page in pages:
        section = page["section"] or section
        seen_on_page: set[str] = set()
        captions = pick_captions(page["items"])
        for k, item in enumerate(page["items"]):
            if item["type"] == "table":
                flush(keep_overlap=False)
                md = scrub(item["markdown"])
                caption = captions.get(k)
                first = md.split("\n")[0]
                cont = ""
                if (not caption and last_table and last_table[0] == page["pdf_page"] - 1
                        and last_table[1] == first):
                    caption = last_table[2]  # same header on the next page = continued table
                    cont = f" (continued from page {last_table[0]})"
                last_table = (page["pdf_page"], first, caption)
                notes = scrub(table_notes(page["items"], k))
                head = f"Section: {section}\n" if section else ""
                if caption:
                    head += f"{scrub(caption)}{cont}\n"
                for part in split_table(md, TABLE_MAX_TOKENS):
                    emit("table", head + part + (f"\n{notes}" if notes else ""),
                         [page["pdf_page"]], section)
                continue

            text = scrub(item["text"])
            if page["needs_vision"]:
                if is_label_noise(text) or text in seen_on_page:
                    continue  # tiny labels; sub-panels repeat the same labels
                seen_on_page.add(text)
            for piece in split_long(text, target):
                blk = {"text": piece, "tokens": ntok(piece),
                       "page": page["pdf_page"], "section": section}
                if buf and sum(b["tokens"] for b in buf) + blk["tokens"] > target:
                    flush(keep_overlap=True)
                    # overlap tail itself must leave room for the new block
                    while buf and sum(b["tokens"] for b in buf) + blk["tokens"] > target:
                        buf.pop(0)
                buf.append(blk)
    flush(keep_overlap=False)

    chunks = merge_tiny(chunks, int(target * 1.25))

    printed = {p["pdf_page"]: p["printed_page"] for p in pages}
    for c in chunks:
        c["printed_pages"] = [printed.get(p) for p in c["pages"]]
        c["tokens"] = ntok(c["text"])
    return chunks


def main(src: str, out_dir: str, target: int, overlap: int) -> None:
    pages = json.loads(Path(src).read_text(encoding="utf-8"))
    chunks = build_chunks(pages, target, overlap)
    doc = pages[0]["doc"]
    for c in chunks:
        c["doc"] = doc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"chunks_{target}.json"
    out_file.write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")

    tok = sorted(c["tokens"] for c in chunks)
    by_type = {t: sum(c["type"] == t for c in chunks) for t in ("text", "table")}
    print(f"target={target} overlap={overlap}: {len(chunks)} chunks {by_type} | "
          f"tokens median={tok[len(tok)//2]} p90={tok[int(len(tok)*.9)]} max={tok[-1]} "
          f"total={sum(tok)}")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    ov = int(sys.argv[4]) if len(sys.argv) > 4 else max(0, int(sys.argv[3]) // 8)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), ov)

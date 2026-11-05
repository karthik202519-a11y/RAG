"""Deterministic real-text corpora of 1M, 3M and 10M characters, built from files that exist on this machine.

Mix (by characters): prose 30% (Markdown, text, the two real PDFs), code 40% (Python, JavaScript, TypeScript, C headers),
data 12% (JSON, CSV), web 8% (HTML), docs-as-code 10% (more Markdown/text). Files are taken in sorted order, so a smaller corpus is a
prefix of a larger one: the same documents, with more haystack around them.
"""
import json
import os
from pathlib import Path

ROOTS = {
    "py": [r"C:\Program Files\Python314\Lib", r"D:\RAG-I\.venv\Lib\site-packages"],
    "js": [r"D:\RAG-I\frontend\node_modules", r"C:\Program Files\nodejs"],
    "md": [r"D:\RAG-I\frontend\node_modules", r"C:\Program Files\nodejs", r"D:\RAG-I\.venv\Lib\site-packages"],
}
MIN_BYTES, MAX_BYTES = 1500, 200_000
BUDGET = {"prose": 0.30, "code": 0.40, "data": 0.12, "web": 0.08, "docs": 0.10}
KIND = {"md": "docs", "txt": "prose", "rst": "prose", "py": "code", "js": "code", "ts": "code", "h": "code",
        "json": "data", "csv": "data", "html": "web", "htm": "web", "xml": "data"}
PDFS = [r"D:\RAG-I\data\studio\artifacts\projects\cc20921189684ca6adfeda416cc7304d\versions\4f30082b54f14f4080a85a32ecf406df\source.pdf",
        r"D:\RAG-I\data\studio\artifacts\projects\cc20921189684ca6adfeda416cc7304d\versions\ab96833489744d9cbdf8246ea9d19819\source.pdf"]


def _walk(roots, exts):
    out = []
    for root in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in ("__pycache__", ".git", "test", "tests", "__tests__", "examples"))
            for f in sorted(fn):
                ext = f.rsplit(".", 1)[-1].lower() if "." in f else ""
                if ext in exts:
                    p = os.path.join(dp, f)
                    try:
                        s = os.path.getsize(p)
                    except OSError:
                        continue
                    if MIN_BYTES <= s <= MAX_BYTES:
                        out.append((p, ext, s))
    return out


def file_pool() -> dict[str, list[tuple[str, str, int]]]:
    pool: dict[str, list] = {k: [] for k in BUDGET}
    for p, ext, s in _walk(ROOTS["py"] + ROOTS["js"], {"py", "js", "ts", "h", "json", "csv", "html", "htm", "xml", "txt", "rst"}):
        pool[KIND[ext]].append((p, ext, s))
    for p, ext, s in _walk(ROOTS["md"], {"md"}):
        pool["docs"].append((p, ext, s))
    # interleave directories so a prefix is varied rather than 'all of one package'
    for k, v in pool.items():
        v.sort(key=lambda t: (hash(os.path.dirname(t[0])) % 997, t[0]))
    return pool


def select(target_chars: int) -> list[tuple[str, str, int]]:
    """Files whose total size is about target_chars, in the BUDGET mix."""
    pool = file_pool()
    picked = []
    for kind, share in BUDGET.items():
        want = target_chars * share * (0.6 if kind == "prose" else 1.0)  # the PDFs supply the rest of the prose share
        got = 0
        for p, ext, s in pool[kind]:
            if got >= want:
                break
            picked.append((p, ext, s))
            got += s
    # The two PDFs supply about 0.9M characters of prose and some pools run dry, so top up from what is left, code first.
    total = sum(x[2] for x in picked) + (900_000 if target_chars >= 2_000_000 else 0)
    chosen = {x[0] for x in picked}
    for kind in ("code", "docs", "data", "web", "prose"):
        for p, ext, s in pool[kind]:
            if total >= target_chars:
                break
            if p not in chosen:
                picked.append((p, ext, s))
                chosen.add(p)
                total += s
    return picked


def load_pdf_pages(path: str, cache_dir: Path) -> list[dict]:
    cache = cache_dir / (Path(path).parent.name + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from studio.parsing import parse_file
    _, pages = parse_file(Path(path), "pdf", Path(path).name, cache_dir / "img")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages), encoding="utf-8")
    return pages


def parsed_documents(target_chars: int, cache_dir: Path) -> list[dict]:
    """[{name, ext, kind, pages}] with parsed pages in the shape every chunker expects."""
    from studio.parsing import parse_file
    docs = []
    for i, p in enumerate(PDFS if target_chars >= 2_000_000 else []):
        if not os.path.exists(p):
            continue
        pages = load_pdf_pages(p, cache_dir)
        chars = sum(len(it.get("text") or it.get("markdown") or "") for pg in pages for it in pg["items"])
        if sum(d["chars"] for d in docs) + chars <= target_chars * 0.5 or not docs:
            docs.append({"name": f"real-{i}.pdf", "ext": "pdf", "kind": "prose", "pages": pages, "chars": chars})
    for path, ext, size in select(target_chars):
        try:
            _, pages = parse_file(Path(path), ext, os.path.basename(path))
        except Exception:
            continue
        chars = sum(len(it.get("text") or it.get("markdown") or "") for pg in pages for it in pg["items"])
        if chars:
            docs.append({"name": os.path.basename(path), "path": path, "ext": ext, "kind": KIND[ext], "pages": pages, "chars": chars})
    return docs

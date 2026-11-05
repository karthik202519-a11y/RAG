"""Entity and relation extraction from one chunk, with every item checked against the chunk text.

The model proposes; this code disposes. An entity is kept only if its name occurs in the chunk, and a relation only if
both of its entities were kept. Anything else is a hallucination and is counted, not stored.
"""
import json
import re

from ..generation import service as gen

SYSTEM = (
    "You extract a knowledge graph from a passage. List the important named entities (systems, documents, reviews, "
    "organisations, roles, standards, processes, quantities with names) exactly as they are written in the passage, "
    "and the relations between them stated in the passage. Do not use outside knowledge. "
    'Reply with JSON only: {"entities":[{"name":"...","type":"..."}],"relations":[{"source":"...","relation":"2-5 words","target":"..."}]}. '
    "At most 12 entities and 12 relations. If the passage has none, return empty lists."
)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_name(s: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise gen.GenerationError("no JSON in reply")
    return json.loads(m.group(0))


def extract_chunk(text: str, model: str, max_output_tokens: int = 900) -> dict:
    """-> {entities: [{name, norm, type}], relations: [{source, relation, target}], dropped, tokens}"""
    res = gen.chat_complete(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text[:6000]}], 0.0, max_output_tokens)
    raw = _parse(res.text)
    body = " " + norm_name(text) + " "
    ents: dict[str, dict] = {}
    dropped = 0
    for e in raw.get("entities") or []:
        name = str(e.get("name") or "").strip()
        n = norm_name(name)
        if len(n) < 2 or len(name) > 120:
            continue
        if f" {n} " in body:
            ents.setdefault(n, {"name": name, "norm": n, "type": str(e.get("type") or "")[:60]})
        else:
            dropped += 1
    rels = []
    for r in raw.get("relations") or []:
        a, b = norm_name(str(r.get("source") or "")), norm_name(str(r.get("target") or ""))
        rel = str(r.get("relation") or "").strip()[:100]
        if a in ents and b in ents and a != b and rel:
            rels.append({"source": a, "relation": rel, "target": b})
        else:
            dropped += 1
    return {"entities": list(ents.values()), "relations": rels, "dropped": dropped, "tokens": res.prompt_tokens + res.completion_tokens}

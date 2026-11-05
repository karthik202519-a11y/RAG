"""LLM judge for answer faithfulness and relevance.

The judge splits an answer into factual claims and, for each, quotes the passage text that supports it. A claim only
counts as *supported* when that quote really occurs in the cited passage (checked here, not trusted from the model).
That makes the score auditable: every verdict carries the evidence it rests on. It is still a model's judgement, so
treat it as a second opinion, not ground truth.
"""
import json
import re

from ..config import settings
from . import service as gen

JUDGE_SYSTEM = (
    "You are a strict fact-checker. You receive a question, numbered context passages, and an answer. "
    "Split the answer into its distinct factual claims (ignore citation markers like [1] and refusals). "
    "For each claim decide: 'supported' if a passage states it, 'contradicted' if a passage says otherwise, "
    "'unsupported' if no passage covers it. For supported or contradicted claims give the passage number and an "
    "exact quote copied word for word from that passage (max 200 characters). "
    "Also rate whether the answer addresses the question: 'yes', 'partial' or 'no'. "
    'Reply with JSON only: {"claims":[{"claim":"...","verdict":"supported|contradicted|unsupported","passage":1,"quote":"..."}],'
    '"relevance":"yes|partial|no"}'
)
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s.lower()).strip()


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise gen.GenerationError("The judge did not return JSON.")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise gen.GenerationError("The judge returned malformed JSON.") from e


def judge_answer(question: str, answer: str, passages: list[dict], model: str | None = None,
                 max_output_tokens: int = 1500) -> dict:
    """passages: [{n, text, ...}] as returned by the answer endpoint."""
    model = model or settings.azure_chat_deployment
    ctx = "\n\n".join(f"[{p['n']}] {p['text']}" for p in passages)
    res = gen.chat_complete(model, [{"role": "system", "content": JUDGE_SYSTEM},
                                    {"role": "user", "content": f"Question: {question}\n\nPassages:\n{ctx}\n\nAnswer:\n{answer}"}],
                            0.0, max_output_tokens)
    raw = _parse(res.text)
    by_n = {p["n"]: _norm(p["text"]) for p in passages}
    claims = []
    for c in raw.get("claims") or []:
        verdict = c.get("verdict") if c.get("verdict") in ("supported", "contradicted", "unsupported") else "unsupported"
        n, quote = c.get("passage"), str(c.get("quote") or "")
        quote_found = bool(quote) and isinstance(n, int) and _norm(quote) in by_n.get(n, "")
        if verdict in ("supported", "contradicted") and not quote_found:
            verdict = "unsupported"  # the judge cited evidence that is not in the passage
        claims.append({"claim": str(c.get("claim", ""))[:500], "verdict": verdict, "passage": n if quote_found else None,
                       "quote": quote[:300] if quote_found else None, "quote_verified": quote_found})
    scored = len(claims)
    sup = sum(c["verdict"] == "supported" for c in claims)
    rel = raw.get("relevance") if raw.get("relevance") in ("yes", "partial", "no") else None
    return {"claims": claims, "faithfulness": round(sup / scored, 3) if scored else None, "supported": sup, "claims_total": scored,
            "contradicted": sum(c["verdict"] == "contradicted" for c in claims), "relevance": rel, "judge_model": res.model,
            "tokens": res.prompt_tokens + res.completion_tokens,
            "note": "Verdicts come from a language model; 'supported' additionally requires the quoted evidence to occur in the passage."}

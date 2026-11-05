"""Contextual embedding: a model writes one sentence that places each chunk in its document, and that sentence is
embedded together with the chunk. The sentence is stored on the chunk so you can read exactly what was added.

Context is built from what is really there: the file name, the section title and the neighbouring text on either side.
The model is told to use only that, and the result is clipped, so a chunk never gains facts that are not in the document.
"""
from ..generation import service as gen

SYSTEM = (
    "You write a short situating sentence for a passage taken from a longer document, to help a search engine find it. "
    "Use only the information provided (file name, section title, neighbouring text, the passage). "
    "Say what the passage is part of and what it is about, in at most 35 words. Do not answer questions, do not add facts "
    "that are not in the text. Reply with the sentence only."
)
MAX_CONTEXT_CHARS = 320


def situate(filename: str, section: str | None, before: str, chunk: str, after: str, model: str) -> tuple[str, int]:
    user = (f"File: {filename}\nSection: {section or 'unknown'}\n\nText before the passage:\n{before[-500:] or '(start of document)'}\n\n"
            f"Passage:\n{chunk[:1500]}\n\nText after the passage:\n{after[:500] or '(end of document)'}")
    res = gen.chat_complete(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], 0.0, 120)
    text = " ".join(res.text.split())[:MAX_CONTEXT_CHARS]
    return text, res.prompt_tokens + res.completion_tokens

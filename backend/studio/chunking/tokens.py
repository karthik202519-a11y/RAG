"""Tokenizers used for chunk sizing. tiktoken encodings are downloaded on first use."""
from functools import lru_cache

import tiktoken

TOKENIZERS = ["cl100k_base", "o200k_base"]  # cl100k = text-embedding-3-*, GPT-4; o200k = GPT-4o family


@lru_cache(maxsize=4)
def get_encoding(name: str) -> tiktoken.Encoding:
    if name not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer '{name}'. Available: {', '.join(TOKENIZERS)}")
    return tiktoken.get_encoding(name)


def count(text: str, name: str = "cl100k_base") -> int:
    return len(get_encoding(name).encode(text, disallowed_special=()))

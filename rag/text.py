"""Tokenisation and chunking shared by the lexical index and ingestion."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has have he her his how i
if in into is it its me more most my no not of on or our she so some such than that the their them then
there these they this those through to too under up was we were what when where which while who whom why
will with would you your about after again against all also am any because before between both during
each few further here just nor off once only other over own same should until very
""".split())


def _stem(t: str) -> str:
    """Plural folding only ("temperatures" -> "temperature"); numbers and short tokens untouched."""
    if len(t) > 3 and t.isalpha() and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-2] if t.endswith("ies") is False and t.endswith("es") and t[-3] in "sxz" else t[:-1]
    return t


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_stem(t) for t in toks]


def chunk_words(text: str, max_words: int = 180, overlap: int = 40) -> list[str]:
    """Split long text into overlapping word windows.

    SQuAD paragraphs are short (median ~120 words), so most become one chunk;
    the window matters for documents ingested through the API.
    """
    if max_words <= overlap:
        raise ValueError("max_words must exceed overlap")
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words) - overlap, step)]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text.strip())
    return [p for p in parts if p]

"""Text normalisation shared by deduplication, grouping and the classifier.

Kept dependency-free so the saved sklearn Pipeline (which references this
function) can be unpickled by the web app without importing training code.
"""
import re

_PLACEHOLDER = re.compile(r"\{\{[^}]+\}\}")
_NON_ALNUM = re.compile(r"[^a-z0-9<> ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, map Bitext {{Placeholders}} to <ph>, drop punctuation, collapse spaces."""
    t = text.lower()
    t = _PLACEHOLDER.sub(" <ph> ", t)
    t = _NON_ALNUM.sub(" ", t)
    return _SPACES.sub(" ", t).strip()

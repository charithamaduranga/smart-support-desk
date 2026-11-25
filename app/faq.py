"""FAQ knowledge base and TF-IDF retrieval (Part B2).

Word-level TF-IDF with English stop words removed: a ticket only scores above
zero against an entry if they share a content word, which makes "no match"
meaningful and the scores easy to explain.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from ml.text import normalize

# Heuristic, not validated: off-topic text shares no content word with any entry
# and scores 0.0; in a 7-query smoke check real matches scored >= 0.09. Below this,
# retrieval means "no FAQ fits" and the LLM is not called. The LLM is a second
# gate: it may still decide the retrieved entries do not answer the ticket.
MIN_SCORE = 0.05


@dataclass(frozen=True)
class FaqEntry:
    id: str
    question: str
    answer: str
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    entry: FaqEntry
    score: float


def load_faq(path: Path) -> list[FaqEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = [FaqEntry(e["id"], e["question"], e["answer"], tuple(e.get("keywords", ())))
               for e in data["entries"]]
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate FAQ ids")
    if not 10 <= len(entries) <= 20 or any(not e.question.strip() or not e.answer.strip() for e in entries):
        raise ValueError("FAQ must have 10-20 entries with non-empty questions and answers")
    return entries


class FaqRetriever:
    def __init__(self, entries: list[FaqEntry]):
        self.entries = entries
        self._vec = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), stop_words="english",
                                    sublinear_tf=True)
        docs = [" ".join([e.question, *e.keywords, e.answer]) for e in entries]
        self._matrix = self._vec.fit_transform(docs)

    def search(self, text: str, k: int = 3, min_score: float = MIN_SCORE) -> list[Hit]:
        scores = (self._vec.transform([text]) @ self._matrix.T).toarray()[0]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [Hit(self.entries[i], round(float(scores[i]), 3)) for i in ranked if scores[i] >= min_score]

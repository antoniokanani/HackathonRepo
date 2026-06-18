"""RAG indexes: TF-IDF, Ollama embeddings, and hybrid retrieval."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import EMBED_MODEL, RagMode
from data import QAPair


def _doc_text(pair: QAPair) -> str:
    return f"{pair.question} {pair.answer}"


class RAGIndex:
    def __init__(
        self,
        train_pairs: list[QAPair],
        top_k: int = 3,
        mode: RagMode = "tfidf",
        embed_model: str = EMBED_MODEL,
    ):
        self.pairs = train_pairs
        self.top_k = top_k
        self.mode = mode
        self.embed_model = embed_model

        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._embeddings: np.ndarray | None = None

        if mode in ("tfidf", "hybrid"):
            self._vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50_000,
            )
            corpus = [_doc_text(p) for p in train_pairs]
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

        if mode in ("embedding", "hybrid"):
            self._embeddings = self._build_embeddings(train_pairs)

    def _build_embeddings(self, train_pairs: list[QAPair], batch_size: int = 32) -> np.ndarray:
        texts = [_doc_text(p) for p in train_pairs]
        vectors: list[list[float]] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding train set"):
            batch = texts[start : start + batch_size]
            response = ollama.embed(model=self.embed_model, input=batch)
            vectors.extend(response["embeddings"])
        return np.array(vectors, dtype=np.float32)

    def _tfidf_scores(self, question: str) -> np.ndarray:
        assert self._vectorizer is not None and self._tfidf_matrix is not None
        query_vec = self._vectorizer.transform([question])
        return cosine_similarity(query_vec, self._tfidf_matrix).flatten()

    def _embedding_scores(self, question: str) -> np.ndarray:
        assert self._embeddings is not None
        response = ollama.embed(model=self.embed_model, input=question)
        query_vec = np.array(response["embeddings"][0], dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = self._embeddings / norms
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(len(self.pairs))
        query_vec = query_vec / query_norm
        return (normalized @ query_vec.T).flatten()

    def _combined_scores(self, question: str) -> np.ndarray:
        tfidf = self._tfidf_scores(question)
        embed = self._embedding_scores(question)
        tfidf = (tfidf - tfidf.min()) / (tfidf.max() - tfidf.min() + 1e-8)
        embed = (embed - embed.min()) / (embed.max() - embed.min() + 1e-8)
        return 0.4 * tfidf + 0.6 * embed

    def retrieve(self, question: str) -> list[QAPair]:
        if self.mode == "tfidf":
            scores = self._tfidf_scores(question)
        elif self.mode == "embedding":
            scores = self._embedding_scores(question)
        else:
            scores = self._combined_scores(question)

        ranked = scores.argsort()[::-1][: self.top_k]
        return [self.pairs[i] for i in ranked]

    def format_context(self, question: str) -> str:
        retrieved = self.retrieve(question)
        blocks = []
        for i, pair in enumerate(retrieved, start=1):
            blocks.append(f"Example {i}:\nQ: {pair.question}\nA: {pair.answer}")
        return "\n\n".join(blocks)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "RAGIndex":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_and_save_index(
    train_pairs: list[QAPair],
    path: Path | str,
    top_k: int = 3,
    mode: RagMode = "tfidf",
    embed_model: str = EMBED_MODEL,
) -> RAGIndex:
    index = RAGIndex(train_pairs, top_k=top_k, mode=mode, embed_model=embed_model)
    index.save(path)
    return index


def save_split_metadata(
    path: Path | str,
    train_count: int,
    eval_count: int,
    seed: int,
    eval_ratio: float,
    extra: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_count": train_count,
        "eval_count": eval_count,
        "seed": seed,
        "eval_ratio": eval_ratio,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

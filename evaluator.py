"""Scoring model answers against reference answers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rouge_score import rouge_scorer


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_key_facts(answer: str) -> list[str]:
    facts = re.findall(r"\*\*([^*]+)\*\*", answer)
    if facts:
        return [f.strip().lower() for f in facts if f.strip()]
    tokens = normalize_text(answer).split()
    return tokens[:5]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens = set(normalize_text(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def key_fact_recall(prediction: str, reference: str) -> float:
    facts = extract_key_facts(reference)
    if not facts:
        return 0.0
    pred_norm = normalize_text(prediction)
    hits = sum(1 for fact in facts if fact in pred_norm)
    return hits / len(facts)


@dataclass
class EvalScores:
    rouge_l: float
    token_f1: float
    key_fact_recall: float

    def average_score(self) -> float:
        return (self.rouge_l + self.token_f1 + self.key_fact_recall) / 3


class Evaluator:
    def __init__(self) -> None:
        self._rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score(self, prediction: str, reference: str) -> EvalScores:
        rouge = self._rouge.score(reference, prediction)["rougeL"].fmeasure
        return EvalScores(
            rouge_l=rouge,
            token_f1=token_f1(prediction, reference),
            key_fact_recall=key_fact_recall(prediction, reference),
        )


@dataclass
class EvalSummary:
    count: int
    avg_rouge_l: float
    avg_token_f1: float
    avg_key_fact_recall: float
    avg_combined: float

    def as_dict(self) -> dict:
        return {
            "count": self.count,
            "avg_rouge_l": round(self.avg_rouge_l, 4),
            "avg_token_f1": round(self.avg_token_f1, 4),
            "avg_key_fact_recall": round(self.avg_key_fact_recall, 4),
            "avg_combined": round(self.avg_combined, 4),
        }


def summarize(scores: list[EvalScores]) -> EvalSummary:
    if not scores:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0)
    n = len(scores)
    return EvalSummary(
        count=n,
        avg_rouge_l=sum(s.rouge_l for s in scores) / n,
        avg_token_f1=sum(s.token_f1 for s in scores) / n,
        avg_key_fact_recall=sum(s.key_fact_recall for s in scores) / n,
        avg_combined=sum(s.average_score() for s in scores) / n,
    )

"""Load and split the World History Q&A dataset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class QAPair:
    question: str
    answer: str


def load_dataset(path: Path | str) -> list[QAPair]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    pairs: list[QAPair] = []
    for item in payload["qa_pairs"]:
        pairs.append(QAPair(question=item["question"], answer=item["answer"]))
    return pairs


def split_dataset(
    pairs: list[QAPair],
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[QAPair], list[QAPair]]:
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, int(len(shuffled) * eval_ratio))
    eval_set = shuffled[:eval_size]
    train_set = shuffled[eval_size:]
    return train_set, eval_set


def iter_batches(items: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

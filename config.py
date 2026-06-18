"""Epoch configurations for progressive Ollama improvements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RagMode = Literal["tfidf", "embedding", "hybrid"]
PromptLevel = Literal["standard", "strict"]

EMBED_MODEL = "nomic-embed-text"

SYSTEM_PROMPTS = {
    "standard": """You are a world history expert for material covered in OpenStax World History Volume 1: to 1500.
Answer clearly and accurately based on the reference context given in each user message.
Prefer short, factual answers. Do not make up dates, names, or events.""",
    "strict": """You are a world history Q&A assistant for OpenStax World History Volume 1: to 1500.

Rules:
1. Answer ONLY using facts present in the reference examples provided in the user message.
2. Include key names, places, and dates from the references when they are relevant.
3. If the references do not contain the answer, respond exactly: "I don't know based on the provided references."
4. Never invent dates, names, events, or speculate beyond the references.
5. Keep answers concise (2-4 sentences) unless the question requires more detail.""",
}

USER_PROMPT_TEMPLATE = """Reference examples from the training dataset:

{context}

Question: {question}

Instructions: Answer using only the references above. Include key names and dates when available."""


@dataclass(frozen=True)
class EpochConfig:
    name: str
    label: str
    base_model: str
    model_name: str
    rag_mode: RagMode
    top_k: int
    temperature: float
    num_ctx: int
    prompt_level: PromptLevel

    def artifact_dir(self, root: str = "artifacts/epochs") -> str:
        return f"{root}/{self.name}"

    def to_dict(self) -> dict:
        return asdict(self)


EPOCHS: list[EpochConfig] = [
    EpochConfig(
        name="epoch_01_baseline",
        label="Epoch 1: Baseline (3B, TF-IDF, top-3)",
        base_model="llama3.2:3b",
        model_name="history-qa-epoch1",
        rag_mode="tfidf",
        top_k=3,
        temperature=0.2,
        num_ctx=8192,
        prompt_level="standard",
    ),
    EpochConfig(
        name="epoch_02_larger_model",
        label="Epoch 2: Larger model (8B, TF-IDF, top-3)",
        base_model="llama3.1:8b",
        model_name="history-qa-epoch2",
        rag_mode="tfidf",
        top_k=3,
        temperature=0.2,
        num_ctx=8192,
        prompt_level="standard",
    ),
    EpochConfig(
        name="epoch_03_more_context",
        label="Epoch 3: More RAG context (8B, TF-IDF, top-5)",
        base_model="llama3.1:8b",
        model_name="history-qa-epoch3",
        rag_mode="tfidf",
        top_k=5,
        temperature=0.15,
        num_ctx=12288,
        prompt_level="standard",
    ),
    EpochConfig(
        name="epoch_04_best",
        label="Epoch 4: Hybrid RAG + strict prompt (8B, top-5)",
        base_model="llama3.1:8b",
        model_name="history-qa-epoch4",
        rag_mode="hybrid",
        top_k=5,
        temperature=0.1,
        num_ctx=16384,
        prompt_level="strict",
    ),
]


def get_epoch(name: str) -> EpochConfig:
    for epoch in EPOCHS:
        if epoch.name == name:
            return epoch
    available = ", ".join(e.name for e in EPOCHS)
    raise ValueError(f"Unknown epoch '{name}'. Available: {available}")

"""Ollama API helpers for setup and inference."""

from __future__ import annotations

import subprocess
from pathlib import Path

import ollama

from config import EMBED_MODEL, SYSTEM_PROMPTS, USER_PROMPT_TEMPLATE, EpochConfig, PromptLevel

DEFAULT_BASE_MODEL = "llama3.2:3b"
CUSTOM_MODEL_NAME = "history-qa"


def pull_model(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def pull_base_model(model: str = DEFAULT_BASE_MODEL) -> None:
    pull_model(model)


def pull_embed_model(model: str = EMBED_MODEL) -> None:
    pull_model(model)


def model_exists(model_name: str) -> bool:
    try:
        ollama.show(model_name)
        return True
    except ollama.ResponseError:
        return False


def ensure_model_exists(model_name: str) -> None:
    if not model_exists(model_name):
        raise RuntimeError(
            f"Ollama model `{model_name}` not found. "
            f"Run setup for this epoch first: python main.py epochs setup --name <epoch_name>"
        )


def create_custom_model(modelfile_path: Path | str, model_name: str) -> None:
    subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        check=True,
    )


def write_modelfile(
    path: Path | str,
    base_model: str = DEFAULT_BASE_MODEL,
    temperature: float = 0.2,
    num_ctx: int = 8192,
    prompt_level: PromptLevel = "standard",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""FROM {base_model}

PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}

SYSTEM \"\"\"{SYSTEM_PROMPTS[prompt_level]}\"\"\"
"""
    path.write_text(content, encoding="utf-8")
    return path


def write_modelfile_for_epoch(path: Path | str, epoch: EpochConfig) -> Path:
    return write_modelfile(
        path,
        base_model=epoch.base_model,
        temperature=epoch.temperature,
        num_ctx=epoch.num_ctx,
        prompt_level=epoch.prompt_level,
    )


def ask(
    question: str,
    context: str,
    model: str = CUSTOM_MODEL_NAME,
    prompt_level: PromptLevel = "standard",
) -> str:
    prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()

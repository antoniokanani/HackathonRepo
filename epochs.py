"""Run progressive epoch setup/eval and compare results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import EPOCHS, EpochConfig
from data import load_dataset, split_dataset
from evaluator import Evaluator, summarize
from ollama_client import (
    ask,
    create_custom_model,
    ensure_model_exists,
    pull_base_model,
    pull_embed_model,
    write_modelfile_for_epoch,
)
from rag import RAGIndex, build_and_save_index, save_split_metadata

ROOT = Path(__file__).resolve().parent
EPOCHS_ROOT = ROOT / "artifacts" / "epochs"
RESULTS_DIR = ROOT / "artifacts" / "results"


def epoch_paths(epoch: EpochConfig) -> dict[str, Path]:
    base = ROOT / epoch.artifact_dir()
    return {
        "dir": base,
        "index": base / "rag_index.pkl",
        "modelfile": base / "Modelfile",
        "meta": base / "epoch_meta.json",
    }


def _results_path(epoch: EpochConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"eval_{epoch.name}_{timestamp}.json"


def setup_epoch(
    epoch: EpochConfig,
    dataset: Path,
    eval_ratio: float,
    seed: int,
) -> None:
    paths = epoch_paths(epoch)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(epoch.label)
    print(f"{'=' * 60}")

    pairs = load_dataset(dataset)
    train, eval_set = split_dataset(pairs, eval_ratio=eval_ratio, seed=seed)
    print(f"Split: {len(train)} train / {len(eval_set)} eval")

    if epoch.rag_mode in ("embedding", "hybrid"):
        print("Pulling embedding model nomic-embed-text...")
        pull_embed_model()

    print(f"Building {epoch.rag_mode} RAG index (top_k={epoch.top_k})...")
    build_and_save_index(
        train,
        paths["index"],
        top_k=epoch.top_k,
        mode=epoch.rag_mode,
    )

    print(f"Pulling base model {epoch.base_model}...")
    pull_base_model(epoch.base_model)

    print(f"Creating Ollama model `{epoch.model_name}`...")
    write_modelfile_for_epoch(paths["modelfile"], epoch)
    create_custom_model(paths["modelfile"], epoch.model_name)

    save_split_metadata(
        paths["meta"],
        len(train),
        len(eval_set),
        seed,
        eval_ratio,
        extra=epoch.to_dict(),
    )
    print(f"Epoch setup saved to {paths['dir']}")


def evaluate_epoch(
    epoch: EpochConfig,
    dataset: Path,
    eval_ratio: float,
    seed: int,
    limit: int | None,
) -> Path:
    paths = epoch_paths(epoch)
    if not paths["index"].exists():
        raise FileNotFoundError(
            f"Index not found for {epoch.name}. Run setup for this epoch first."
        )

    pairs = load_dataset(dataset)
    _, eval_set = split_dataset(pairs, eval_ratio=eval_ratio, seed=seed)
    if limit:
        eval_set = eval_set[:limit]

    index = RAGIndex.load(paths["index"])
    evaluator = Evaluator()
    ensure_model_exists(epoch.model_name)

    print(f"\nEvaluating {len(eval_set)} questions with `{epoch.model_name}`...")
    results = []
    scores = []
    errors: list[str] = []

    for i, pair in enumerate(eval_set, start=1):
        context = index.format_context(pair.question)
        try:
            prediction = ask(
                pair.question,
                context,
                model=epoch.model_name,
                prompt_level=epoch.prompt_level,
            )
        except Exception as exc:
            msg = f"[{i}/{len(eval_set)}] {exc}"
            errors.append(msg)
            print(f"  ERROR: {msg}")
            continue

        score = evaluator.score(prediction, pair.answer)
        scores.append(score)
        results.append(
            {
                "question": pair.question,
                "reference": pair.answer,
                "prediction": prediction,
                "scores": {
                    "rouge_l": round(score.rouge_l, 4),
                    "token_f1": round(score.token_f1, 4),
                    "key_fact_recall": round(score.key_fact_recall, 4),
                    "combined": round(score.average_score(), 4),
                },
            }
        )
        print(
            f"  [{i}/{len(eval_set)}] combined={score.average_score():.3f} "
            f"rouge={score.rouge_l:.3f} f1={score.token_f1:.3f} facts={score.key_fact_recall:.3f}"
        )

    if not results:
        hint = errors[0] if errors else "unknown error"
        raise RuntimeError(
            f"No questions were evaluated for {epoch.name}. "
            f"Setup may be incomplete. First error: {hint}"
        )

    summary = summarize(scores)
    payload = {
        "model": epoch.model_name,
        "epoch": epoch.to_dict(),
        "eval_count": len(results),
        "evaluated_at": datetime.now().isoformat(),
        "error_count": len(errors),
        "summary": summary.as_dict(),
        "results": results,
    }
    out_path = _results_path(epoch)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n--- Epoch summary ---")
    print(f"  Combined score:  {summary.avg_combined:.4f}")
    print(f"  Key-fact recall: {summary.avg_key_fact_recall:.4f}")
    print(f"  Results:         {out_path}")
    return out_path


def run_all_epochs(
    dataset: Path,
    eval_ratio: float,
    seed: int,
    limit: int | None,
    start: int = 1,
    end: int | None = None,
) -> list[Path]:
    selected = EPOCHS[start - 1 : end]
    result_paths: list[Path] = []

    for epoch in selected:
        setup_epoch(epoch, dataset, eval_ratio, seed)
        result_paths.append(evaluate_epoch(epoch, dataset, eval_ratio, seed, limit))

    manifest = {
        "run_at": datetime.now().isoformat(),
        "epochs": [epoch.to_dict() for epoch in selected],
        "result_files": [str(p) for p in result_paths],
    }
    manifest_path = RESULTS_DIR / "epochs_manifest.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll epochs complete. Manifest: {manifest_path}")
    return result_paths


def load_epoch_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []

    latest_by_epoch: dict[str, tuple[str, dict]] = {}
    for path in sorted(RESULTS_DIR.glob("eval_*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        epoch_info = data.get("epoch")
        if not epoch_info:
            continue
        name = epoch_info["name"]
        mtime = path.stat().st_mtime
        if name not in latest_by_epoch or mtime > latest_by_epoch[name][0]:
            latest_by_epoch[name] = (mtime, data)

    epoch_order = {e.name: i for i, e in enumerate(EPOCHS)}
    return [
        latest_by_epoch[name][1]
        for name in sorted(latest_by_epoch, key=lambda n: epoch_order.get(n, 999))
    ]


def compare_epochs() -> None:
    rows = load_epoch_results()
    if not rows:
        print("No epoch results found. Run: python main.py epochs run")
        return

    print("\n=== Epoch Comparison ===\n")
    header = f"{'Epoch':<28} {'Combined':>10} {'ROUGE-L':>10} {'Token F1':>10} {'Key Facts':>10} {'Δ Combined':>12}"
    print(header)
    print("-" * len(header))

    prev_combined: float | None = None
    for row in rows:
        epoch = row["epoch"]
        summary = row["summary"]
        combined = summary["avg_combined"]
        delta = ""
        if prev_combined is not None:
            change = combined - prev_combined
            sign = "+" if change >= 0 else ""
            delta = f"{sign}{change:.4f}"
        prev_combined = combined

        print(
            f"{epoch['name']:<28} "
            f"{summary['avg_combined']:>10.4f} "
            f"{summary['avg_rouge_l']:>10.4f} "
            f"{summary['avg_token_f1']:>10.4f} "
            f"{summary['avg_key_fact_recall']:>10.4f} "
            f"{delta:>12}"
        )

    best = max(rows, key=lambda r: r["summary"]["avg_combined"])
    print(f"\nBest epoch: {best['epoch']['name']} (combined={best['summary']['avg_combined']:.4f})")
    print(f"Model:      {best['epoch']['model_name']}")


def list_epochs() -> None:
    print("\n=== Training Epochs ===\n")
    for i, epoch in enumerate(EPOCHS, start=1):
        print(f"Epoch {i}: {epoch.label}")
        print(f"  Model:       {epoch.model_name} (from {epoch.base_model})")
        print(f"  RAG:         {epoch.rag_mode}, top_k={epoch.top_k}")
        print(f"  Temperature: {epoch.temperature}, num_ctx={epoch.num_ctx}")
        print(f"  Prompt:      {epoch.prompt_level}")
        print()

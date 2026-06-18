"""CLI for setting up Ollama + RAG and evaluating Q&A accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from config import EPOCHS, get_epoch
from data import load_dataset, split_dataset
from epochs import (
    compare_epochs,
    evaluate_epoch,
    list_epochs,
    run_all_epochs,
    setup_epoch,
)
from evaluator import Evaluator, summarize
from ollama_client import (
    CUSTOM_MODEL_NAME,
    DEFAULT_BASE_MODEL,
    ask,
    create_custom_model,
    pull_base_model,
    write_modelfile,
)
from rag import RAGIndex, build_and_save_index, save_split_metadata

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset.json"
ARTIFACTS = ROOT / "artifacts"
INDEX_PATH = ARTIFACTS / "rag_index.pkl"
MODELFILE_PATH = ARTIFACTS / "Modelfile"
SPLIT_META_PATH = ARTIFACTS / "split_meta.json"
RESULTS_DIR = ARTIFACTS / "results"


def cmd_setup(args: argparse.Namespace) -> None:
    print(f"Loading dataset from {args.dataset}...")
    pairs = load_dataset(args.dataset)
    train, eval_set = split_dataset(pairs, eval_ratio=args.eval_ratio, seed=args.seed)
    print(f"Split: {len(train)} train / {len(eval_set)} eval")

    print("Building RAG index from training data...")
    build_and_save_index(train, INDEX_PATH, top_k=args.top_k, mode=args.rag_mode)
    save_split_metadata(
        SPLIT_META_PATH,
        len(train),
        len(eval_set),
        args.seed,
        args.eval_ratio,
        extra={"rag_mode": args.rag_mode, "top_k": args.top_k},
    )

    print(f"Pulling base model {args.base_model} (this may take a few minutes)...")
    pull_base_model(args.base_model)

    print("Creating custom Ollama model...")
    write_modelfile(
        MODELFILE_PATH,
        base_model=args.base_model,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        prompt_level=args.prompt_level,
    )
    create_custom_model(MODELFILE_PATH, model_name=args.model)

    print("\nSetup complete.")
    print(f"  Custom model: {args.model}")
    print(f"  RAG index:    {INDEX_PATH}")
    print(f"  Eval set size: {len(eval_set)}")
    print(f"\nRun evaluation with:\n  python main.py evaluate --limit {args.limit or 20}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    if not INDEX_PATH.exists():
        print("RAG index not found. Run `python main.py setup` first.", file=sys.stderr)
        sys.exit(1)

    pairs = load_dataset(args.dataset)
    _, eval_set = split_dataset(pairs, eval_ratio=args.eval_ratio, seed=args.seed)
    if args.limit:
        eval_set = eval_set[: args.limit]

    index = RAGIndex.load(INDEX_PATH)
    evaluator = Evaluator()

    print(f"Evaluating {len(eval_set)} questions with model `{args.model}`...")
    results = []
    scores = []

    for i, pair in enumerate(eval_set, start=1):
        context = index.format_context(pair.question)
        try:
            prediction = ask(
                pair.question,
                context,
                model=args.model,
                prompt_level=args.prompt_level,
            )
        except Exception as exc:
            print(f"  [{i}/{len(eval_set)}] ERROR: {exc}", file=sys.stderr)
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

    summary = summarize(scores)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    payload = {
        "model": args.model,
        "eval_count": len(results),
        "summary": summary.as_dict(),
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Evaluation Summary ===")
    print(f"  Questions evaluated: {summary.count}")
    print(f"  Avg ROUGE-L:         {summary.avg_rouge_l:.4f}")
    print(f"  Avg token F1:        {summary.avg_token_f1:.4f}")
    print(f"  Avg key-fact recall: {summary.avg_key_fact_recall:.4f}")
    print(f"  Avg combined score:  {summary.avg_combined:.4f}")
    print(f"\nDetailed results saved to {out_path}")


def cmd_epochs_run(args: argparse.Namespace) -> None:
    run_all_epochs(
        dataset=args.dataset,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        limit=args.limit,
        start=args.start,
        end=args.end,
    )
    compare_epochs()


def cmd_epochs_setup(args: argparse.Namespace) -> None:
    epoch = get_epoch(args.name)
    setup_epoch(epoch, args.dataset, args.eval_ratio, args.seed)


def cmd_epochs_evaluate(args: argparse.Namespace) -> None:
    epoch = get_epoch(args.name)
    evaluate_epoch(epoch, args.dataset, args.eval_ratio, args.seed, args.limit)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train (RAG) and evaluate a local Ollama model on World History Q&A."
    )
    add_shared_args(parser)

    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="Pull model, build RAG index, create custom Ollama model")
    setup.add_argument("--model", default=CUSTOM_MODEL_NAME)
    setup.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    setup.add_argument("--top-k", type=int, default=3)
    setup.add_argument("--rag-mode", choices=["tfidf", "embedding", "hybrid"], default="tfidf")
    setup.add_argument("--temperature", type=float, default=0.2)
    setup.add_argument("--num-ctx", type=int, default=8192)
    setup.add_argument("--prompt-level", choices=["standard", "strict"], default="standard")
    setup.add_argument("--limit", type=int, default=None)
    setup.set_defaults(func=cmd_setup)

    evaluate = sub.add_parser("evaluate", help="Run evaluation on held-out questions")
    evaluate.add_argument("--model", default=CUSTOM_MODEL_NAME)
    evaluate.add_argument("--prompt-level", choices=["standard", "strict"], default="standard")
    evaluate.add_argument("--limit", type=int, default=20, help="Use 0 for all eval questions")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = sub.add_parser("compare", help="Compare results across all completed epochs")
    compare.set_defaults(func=lambda args: compare_epochs())

    epochs = sub.add_parser("epochs", help="Progressive improvement across 4 epochs")
    epochs_sub = epochs.add_subparsers(dest="epochs_command", required=True)

    epochs_list = epochs_sub.add_parser("list", help="Show all epoch configurations")
    epochs_list.set_defaults(func=lambda args: list_epochs())

    epochs_run = epochs_sub.add_parser("run", help="Run all epochs (setup + evaluate each)")
    epochs_run.add_argument("--limit", type=int, default=20, help="Use 0 for all eval questions")
    epochs_run.add_argument("--start", type=int, default=1, help="First epoch number (1-4)")
    epochs_run.add_argument("--end", type=int, default=None, help="Last epoch number (1-4)")
    epochs_run.set_defaults(func=cmd_epochs_run)

    epochs_setup = epochs_sub.add_parser("setup", help="Setup a single epoch by name")
    epochs_setup.add_argument(
        "--name",
        required=True,
        choices=[e.name for e in EPOCHS],
        help="Epoch name",
    )
    epochs_setup.set_defaults(func=cmd_epochs_setup)

    epochs_eval = epochs_sub.add_parser("evaluate", help="Evaluate a single epoch by name")
    epochs_eval.add_argument(
        "--name",
        required=True,
        choices=[e.name for e in EPOCHS],
        help="Epoch name",
    )
    epochs_eval.add_argument("--limit", type=int, default=20, help="Use 0 for all eval questions")
    epochs_eval.set_defaults(func=cmd_epochs_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) == "evaluate" and args.limit == 0:
        args.limit = None
    if getattr(args, "epochs_command", None) == "evaluate" and args.limit == 0:
        args.limit = None
    if getattr(args, "epochs_command", None) == "run" and args.limit == 0:
        args.limit = None
    args.func(args)


if __name__ == "__main__":
    main()

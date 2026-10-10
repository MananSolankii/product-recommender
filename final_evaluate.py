"""Lesson 8: one final held-out test evaluation using Step 7's fixed choices."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from benchmark_movielens import ARCHIVE, download, load_ratings
from content_model import ContentRecommender, METADATA_FILE, load_metadata
from evaluate import evaluate, split_by_time
from explore_data import DATA_FILE, prepare_reviews
from neighbor_model import NeighborRecommender

ROOT = Path(__file__).resolve().parent
EXPECTED_CONFIGS = {
    "movielens": {"name": "neighbors_10", "kind": "neighbors", "neighbors": 10},
    "products": {"name": "content_tfidf", "kind": "content_tfidf"},
}


def sha256(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def load_fixed_selection(dataset):
    """Load and validate Step 7's saved choice; do not choose from test results."""
    filename = ROOT / f"artifacts/lesson7_{dataset}_selection.json"
    if not filename.exists():
        raise SystemExit(
            f"Missing {filename.name}. Run python tune_models.py --dataset {dataset} first."
        )
    try:
        report = json.loads(filename.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Cannot read {filename.name}: {error}") from error
    if report.get("stage") != "validation_selection" or report.get("dataset") != dataset:
        raise SystemExit(f"{filename.name} is not the expected Step 7 selection report.")
    config = report.get("selected_config")
    if config != EXPECTED_CONFIGS[dataset]:
        raise SystemExit(
            f"{filename.name} does not contain the fixed expected selection. "
            "Do not change settings after seeing test data; investigate the earlier run."
        )
    return filename, report


def verify_inputs(selection, inputs):
    expected = selection.get("input_sha256", {})
    actual = {Path(item).name: sha256(item) for item in inputs}
    if expected != actual:
        raise SystemExit(
            "Current input file hashes differ from the data used for Step 7 selection. "
            "Do not mix a selection report with changed data. Re-run the validation process "
            "and record why the data changed before a new final assessment."
        )
    return actual


def format_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def print_results(results):
    print(f"{'Model':<18} {'Group':<18} {'Users':>6} {'Recall@5':>10} {'NDCG@5':>10}")
    for name, result in results.items():
        for group in ("all_users", "returning_users", "new_users"):
            row = result[group]
            print(f"{name:<18} {group:<18} {row['evaluated_users']:>6} "
                  f"{format_metric(row['recall_at_5']):>10} "
                  f"{format_metric(row['ndcg_at_5']):>10}")


def evaluate_dataset(dataset):
    selection_path, selection = load_fixed_selection(dataset)
    if dataset == "movielens":
        download()
        reviews = load_ratings(ARCHIVE)
        metadata = None
        inputs = [ARCHIVE]
    else:
        if not DATA_FILE.exists() or not METADATA_FILE.exists():
            raise SystemExit("Product data is missing. Run the earlier data-download steps first.")
        reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
        metadata = load_metadata()
        inputs = [DATA_FILE, METADATA_FILE]
    input_hashes = verify_inputs(selection, inputs)
    train, validation, test = split_by_time(reviews)
    history = pd.concat([train, validation], ignore_index=True)
    config = selection["selected_config"]
    if config["kind"] == "neighbors":
        model = NeighborRecommender(history, neighbors=config["neighbors"])
    elif config["kind"] == "content_tfidf":
        model = ContentRecommender(history, metadata)
    else:  # Defensive: only fixed configurations are accepted above.
        raise RuntimeError("Unsupported fixed configuration.")
    results = {
        "popularity": evaluate(history, test),
        config["name"]: evaluate(history, test, recommend_fn=model.recommend),
    }
    print(f"\n{dataset.upper()}: FINAL HELD-OUT TEST RESULTS")
    print("Selected before this run:", config["name"])
    print_results(results)
    print("No settings were changed or selected using these test results.")
    if dataset == "products":
        print("Caveat: TF-IDF uses a later snapshot of product descriptions.")
    report = {
        "stage": "final_held_out_test",
        "dataset": dataset,
        "selected_before_test": config,
        "selection_report": selection_path.name,
        "selection_report_sha256": sha256(selection_path),
        "input_sha256": input_hashes,
        "results": results,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "split": {
            "train": {"rows": len(train), "start": str(train.timestamp.min()),
                      "end": str(train.timestamp.max())},
            "validation": {"rows": len(validation), "start": str(validation.timestamp.min()),
                           "end": str(validation.timestamp.max())},
            "refit_history": {"rows": len(history), "start": str(history.timestamp.min()),
                              "end": str(history.timestamp.max())},
            "test": {"rows": len(test), "start": str(test.timestamp.min()),
                     "end": str(test.timestamp.max())},
        },
        "limitations": [
            "A single held-out period is not a statistical-significance or business-impact claim.",
            "MovieLens and product scores are separate tasks and are not directly comparable.",
            "Product results have only a small returning-user subgroup.",
            "The product TF-IDF model uses later metadata descriptions.",
        ],
    }
    output = ROOT / f"artifacts/lesson8_{dataset}_test.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["movielens", "products", "both"], default="both")
    args = parser.parse_args()
    for dataset in (("movielens", "products") if args.dataset == "both" else (args.dataset,)):
        evaluate_dataset(dataset)


if __name__ == "__main__":
    main()

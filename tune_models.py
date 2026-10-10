"""Lesson 7: fixed validation experiments; select separately for each dataset."""
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
from collaborative_model import CollaborativeRecommender
from content_model import ContentRecommender, METADATA_FILE, load_metadata
from evaluate import evaluate, split_by_time
from explore_data import DATA_FILE, prepare_reviews
from neighbor_model import NeighborRecommender

ROOT = Path(__file__).resolve().parent
# Decided before running the new experiments. Do not keep expanding this grid.
NEIGHBOR_COUNTS = (10, 30, 100)
SELECTION_RULE = (
    "Maximize all_users NDCG@5, then all_users Recall@5; "
    "exact ties prefer the first candidate in the recorded order."
)


def select_best(candidates):
    """Candidates are ordered; max keeps the first on an exact metric tie."""
    if not candidates or any(
        c["metrics"]["all_users"]["ndcg_at_5"] is None for c in candidates
    ):
        raise ValueError("Selection needs candidates with eligible validation users.")
    return max(candidates, key=lambda candidate: (
        candidate["metrics"]["all_users"]["ndcg_at_5"],
        candidate["metrics"]["all_users"]["recall_at_5"],
    ))


def sha256(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def run_experiment(dataset):
    metadata = None
    if dataset == "movielens":
        download()
        reviews = load_ratings(ARCHIVE)
        inputs = [ARCHIVE]
    else:
        if not DATA_FILE.exists() or not METADATA_FILE.exists():
            raise SystemExit(
                "Product data is missing. Run download_data.py and download_metadata.py first."
            )
        reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
        metadata = load_metadata()
        inputs = [DATA_FILE, METADATA_FILE]
    train, validation, reserved_test = split_by_time(reviews)
    # Nothing below fits or scores using reserved_test.
    configs = [{"name": "popularity", "kind": "popularity"}]
    if dataset == "products":
        configs.append({"name": "content_tfidf", "kind": "content_tfidf"})
    configs.append({"name": "collaborative", "kind": "collaborative"})
    configs.extend(
        {"name": f"neighbors_{n}", "kind": "neighbors", "neighbors": n}
        for n in NEIGHBOR_COUNTS
    )
    print(f"\n{dataset.upper()}: VALIDATION ONLY")
    print("Selection:", SELECTION_RULE)
    print("Fixed candidate order:", ", ".join(c["name"] for c in configs))
    print(f"{'Model':<17} {'All Recall':>11} {'All NDCG':>10} "
          f"{'Return Recall':>14} {'Return NDCG':>12} {'Links':>9}")
    candidates = []
    for config in configs:
        kind = config["kind"]
        model = None
        if kind == "content_tfidf":
            model = ContentRecommender(train, metadata)
        elif kind == "collaborative":
            model = CollaborativeRecommender(train)
        elif kind == "neighbors":
            model = NeighborRecommender(train, neighbors=config["neighbors"])
        result = evaluate(train, validation, recommend_fn=model.recommend if model else None)
        entry = {"config": config, "metrics": result}
        links = "-"
        if kind in ("collaborative", "neighbors"):
            links = str(model.similarity.nnz)
            entry["directed_similarity_links"] = int(model.similarity.nnz)
        candidates.append(entry)
        all_row, returning = result["all_users"], result["returning_users"]
        def display(value):
            return "n/a" if value is None else f"{value:.4f}"
        print(f"{config['name']:<17} {display(all_row['recall_at_5']):>11} "
              f"{display(all_row['ndcg_at_5']):>10} "
              f"{display(returning['recall_at_5']):>14} "
              f"{display(returning['ndcg_at_5']):>12} {links:>9}")
    best = select_best(candidates)
    counts = candidates[0]["metrics"]
    print(f"Eligible users: {counts['all_users']['evaluated_users']}; "
          f"returning: {counts['returning_users']['evaluated_users']}; "
          f"new: {counts['new_users']['evaluated_users']}")
    print("Selected for final assessment:", best["config"]["name"])
    if dataset == "products":
        print("TF-IDF caveat: product descriptions come from a later metadata snapshot.")
    print("Validation-selected, not a proven general winner. No test scores computed.")
    sources = [
        "tune_models.py", "neighbor_model.py", "collaborative_model.py",
        "evaluate.py", "explore_data.py", "content_model.py", "benchmark_movielens.py",
    ]
    report = {
        "stage": "validation_selection",
        "dataset": dataset,
        "selection_rule": SELECTION_RULE,
        "neighbor_grid": list(NEIGHBOR_COUNTS),
        "selected_config": best["config"],
        "candidates": candidates,
        "input_sha256": {p.name: sha256(p) for p in inputs},
        "code_sha256": {name: sha256(ROOT / name) for name in sources},
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "split": {
            name: {"rows": len(part), "start": str(part.timestamp.min()),
                   "end": str(part.timestamp.max())}
            for name, part in (("train", train), ("validation", validation),
                               ("test", reserved_test))
        },
        "limitations": [
            "Repeated validation exploration can overfit; keep this small grid fixed.",
            "No statistical significance or business-impact claim.",
            "MovieLens and product metrics describe separate tasks.",
            "Product TF-IDF uses later snapshot descriptions, not verified historical text.",
        ],
    }
    output = ROOT / f"artifacts/lesson7_{dataset}_selection.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["movielens", "products", "both"], default="both")
    args = parser.parse_args()
    for dataset in (("movielens", "products") if args.dataset == "both" else (args.dataset,)):
        run_experiment(dataset)


if __name__ == "__main__":
    main()

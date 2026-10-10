"""Compare popularity and TF-IDF using the same Lesson 3 validation protocol."""
import json
from pathlib import Path
import pandas as pd

from content_model import ContentRecommender, load_metadata
from evaluate import evaluate, split_by_time
from explore_data import DATA_FILE, prepare_reviews


def main():
    if not DATA_FILE.exists():
        raise SystemExit("Run python download_data.py first.")
    reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
    train, validation, _ = split_by_time(reviews)
    model = ContentRecommender(train, load_metadata())
    baseline = evaluate(train, validation)
    content = evaluate(train, validation, recommend_fn=model.recommend)
    report = {
        "stage": "validation",
        "metadata_caveat": (
            "Descriptions are a crawl-time snapshot, not verified historical versions. "
            "This is a retrospective content experiment, not a fully time-faithful backtest."
        ),
        "popularity": baseline,
        "tfidf_with_popularity_fallback": content,
    }
    print("VALIDATION COMPARISON (K=5)")
    print(f"{'Group':<18} {'Users':>6} {'Pop Recall':>12} {'TFIDF Recall':>13}"
          f" {'Pop NDCG':>10} {'TFIDF NDCG':>11}")
    for group in ("all_users", "returning_users", "new_users"):
        a, b = baseline[group], content[group]
        def score(value):
            return "n/a" if value is None else f"{value:.4f}"
        print(f"{group:<18} {a['evaluated_users']:>6}"
              f" {score(a['recall_at_5']):>12} {score(b['recall_at_5']):>13}"
              f" {score(a['ndcg_at_5']):>10} {score(b['ndcg_at_5']):>11}")
    print("\n" + report["metadata_caveat"])
    print("New users use the same popularity fallback in both approaches.")
    print("Test scores are still reserved. Better scores are not guaranteed.")
    output = Path(__file__).resolve().parent / "artifacts/lesson4_comparison.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

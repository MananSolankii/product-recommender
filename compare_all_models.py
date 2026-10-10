"""Lesson 5: compare popularity, content, and collaborative recommendations."""
import inspect
import json
from pathlib import Path
import pandas as pd

from collaborative_model import CollaborativeRecommender
from content_model import ContentRecommender, load_metadata
from evaluate import evaluate, split_by_time
from explore_data import DATA_FILE, prepare_reviews


def main():
    if "recommend_fn" not in inspect.signature(evaluate).parameters:
        raise SystemExit("Your evaluate.py is outdated. Copy evaluate.py from the updated ZIP.")
    if not DATA_FILE.exists():
        raise SystemExit("Run python download_data.py first.")
    reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
    train, validation, _ = split_by_time(reviews)
    content = ContentRecommender(train, load_metadata())
    collaborative = CollaborativeRecommender(train)
    models = {
        "popularity": None,
        "content_tfidf": content.recommend,
        "collaborative": collaborative.recommend,
    }
    results = {}
    print("VALIDATION ONLY: SAME SPLIT, CATALOG, USERS AND K=5")
    print(f"{'Model':<16} {'Group':<18} {'Users':>6} {'Recall@5':>10} {'NDCG@5':>10}")
    for name, recommender in models.items():
        result = evaluate(train, validation, recommend_fn=recommender)
        results[name] = result
        for group in ("all_users", "returning_users", "new_users"):
            row = result[group]
            recall = "n/a" if row["recall_at_5"] is None else f"{row['recall_at_5']:.4f}"
            ndcg = "n/a" if row["ndcg_at_5"] is None else f"{row['ndcg_at_5']:.4f}"
            print(f"{name:<16} {group:<18} {row['evaluated_users']:>6} {recall:>10} {ndcg:>10}")

    # Report the number who actually have usable collaborative evidence.
    eligible = []
    for user, events in validation.groupby("user_id"):
        targets = set(events.loc[events["rating"] >= 4, "product_id"])
        targets -= collaborative.seen.get(user, set())
        if targets:
            eligible.append(user)
    signal_users = sum(collaborative.has_personal_signal(user) for user in eligible)
    diagnostics = {
        **collaborative.training_summary,
        "evaluated_users_with_unseen_collaborative_signal": signal_users,
        "evaluated_users_using_only_popularity_fallback": len(eligible) - signal_users,
    }
    print("\nCOLLABORATIVE DATA DIAGNOSTICS")
    print(json.dumps(diagnostics, indent=2))
    returning_count = results["popularity"]["returning_users"]["evaluated_users"]
    print(f"\nReturning users evaluated: {returning_count}. Interpret scores with this count.")
    print("CF uses historical reviews only; TF-IDF has the later-description snapshot limitation.")
    print("No test scores were computed. Do not claim this comparison establishes a general winner.")
    report = {
        "stage": "validation",
        "models": results,
        "collaborative_diagnostics": diagnostics,
        "metadata_caveat": "TF-IDF uses later crawl-time product descriptions; CF does not use metadata.",
    }
    output = Path(__file__).resolve().parent / "artifacts/lesson5_comparison.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

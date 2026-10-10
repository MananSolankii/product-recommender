"""Lesson 3: chronological evaluation of the popularity baseline."""
import argparse
import json
from math import log2
from pathlib import Path

import pandas as pd
from explore_data import DATA_FILE, prepare_reviews

ROOT = Path(__file__).resolve().parent


def split_by_time(reviews):
    """Approximate 70/15/15 split; keep equal timestamps in the same period."""
    ordered = reviews.sort_values("timestamp").reset_index(drop=True)
    if len(ordered) < 3:
        raise ValueError("Need at least three reviews.")
    first_cut = ordered.iloc[int(len(ordered) * 0.70)]["timestamp"]
    second_cut = ordered.iloc[int(len(ordered) * 0.85)]["timestamp"]
    train = ordered.loc[ordered["timestamp"] < first_cut].copy()
    validation = ordered.loc[
        (ordered["timestamp"] >= first_cut) & (ordered["timestamp"] < second_cut)
    ].copy()
    test = ordered.loc[ordered["timestamp"] >= second_cut].copy()
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("Not enough distinct times for three nonempty periods.")
    return train, validation, test


def fit_popularity(history):
    """Learn counts from history only, using one positive vote per user/item."""
    positive = history.loc[history["rating"] >= 4]
    counts = positive.groupby("product_id")["user_id"].nunique().to_dict()
    catalog = set(history["product_id"])
    # Zero-positive products are allowed at the bottom as a fallback.
    ranking = sorted(catalog, key=lambda product: (-counts.get(product, 0), product))
    seen = history.groupby("user_id")["product_id"].agg(set).to_dict()
    return ranking, seen


def ranking_metrics(recommendations, relevant, k):
    """Binary Recall@K and NDCG@K for one user with nonempty targets."""
    if k < 1 or not relevant:
        raise ValueError("Use k >= 1 and at least one relevant product.")
    top = recommendations[:k]
    if len(set(top)) != len(top):
        raise ValueError("Recommendations must not repeat products.")
    hits = len(set(top) & relevant)
    recall = hits / len(relevant)
    dcg = sum(
        1 / log2(rank + 1)
        for rank, product in enumerate(top, start=1) if product in relevant
    )
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return recall, dcg / ideal


def evaluate(history, future, k=5, recommend_fn=None):
    """Freeze recommendations at the boundary, then compare with future likes."""
    if k < 1:
        raise ValueError("k must be positive.")
    if history.empty or future.empty:
        raise ValueError("History and future must both be nonempty.")
    if history["timestamp"].max() >= future["timestamp"].min():
        raise ValueError("History must be strictly earlier than evaluation data.")
    ranking, seen = fit_popularity(history)
    catalog = set(ranking)
    rows = []
    excluded = 0
    for user_id, events in future.groupby("user_id", sort=True):
        previously_seen = seen.get(user_id, set())
        relevant = set(events.loc[events["rating"] >= 4, "product_id"]) - previously_seen
        if not relevant:
            excluded += 1
            continue
        recommendations = (recommend_fn(user_id, k) if recommend_fn is not None
                           else [p for p in ranking if p not in previously_seen][:k])
        if len(recommendations) > k or any(
            p not in catalog or p in previously_seen for p in recommendations
        ):
            raise ValueError("Recommend at most K unseen historical products.")
        recall, ndcg = ranking_metrics(recommendations, relevant, k)
        rows.append({
            "group": "returning_users" if user_id in seen else "new_users",
            "recall": recall,
            "ndcg": ndcg,
            "targets": len(relevant),
            # Keep these targets in the denominator: we cannot predict new items.
            "unknown_targets": len(relevant - catalog),
        })

    def aggregate(selected):
        return {
            "evaluated_users": len(selected),
            f"recall_at_{k}": (sum(r["recall"] for r in selected) / len(selected)
                              if selected else None),
            f"ndcg_at_{k}": (sum(r["ndcg"] for r in selected) / len(selected)
                            if selected else None),
            "relevant_user_product_pairs": sum(r["targets"] for r in selected),
            "targets_absent_from_history_catalog": sum(r["unknown_targets"] for r in selected),
        }

    result = {
        "k": k,
        "users_in_future_period": int(future["user_id"].nunique()),
        "users_without_new_positive_targets_excluded": excluded,
        "history_catalog_products": len(catalog),
        "all_users": aggregate(rows),
    }
    for group in ("returning_users", "new_users"):
        result[group] = aggregate([r for r in rows if r["group"] == group])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true",
                        help="Final test: run only after model choices are settled.")
    args = parser.parse_args()
    if not DATA_FILE.exists():
        raise SystemExit("Run python download_data.py first.")
    raw = pd.read_json(DATA_FILE, lines=True, convert_dates=False)
    reviews, _ = prepare_reviews(raw)
    train, validation, test = split_by_time(reviews)
    split_info = {
        name: {"rows": len(part), "start": str(part["timestamp"].min()),
               "end": str(part["timestamp"].max())}
        for name, part in (("train", train), ("validation", validation), ("test", test))
    }
    stage = "test" if args.test else "validation"
    history = pd.concat([train, validation], ignore_index=True) if args.test else train
    future = test if args.test else validation
    results = evaluate(history, future, k=5)
    report = {"stage": stage, "split": split_info, "results": results}
    print(f"LESSON 3: {stage.upper()} RESULTS")
    print(json.dumps(report, indent=2))
    print("\nScores are per-user averages over users with new positive targets.")
    print("Unreviewed products are unobserved preferences, not proven dislikes.")
    if not args.test:
        print("Test ratings were not used for fitting or scoring; keep test for final assessment.")
    print("This is an offline popularity baseline, not evidence of personalization.")
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    destination = output / f"lesson3_{stage}.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

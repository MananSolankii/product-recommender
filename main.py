"""Lesson 1: an explainable popularity baseline using fictional practice data."""

import argparse
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "practice_reviews.json"


def load_data(path):
    """Read the JSON file and catch common mistakes in our practice data."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    products = data["products"]
    reviews = data["reviews"]
    seen_pairs = set()
    for review in reviews:
        if review["product_id"] not in products:
            raise ValueError("A review refers to an unknown product.")
        if not 1 <= review["rating"] <= 5:
            raise ValueError("Ratings must be between 1 and 5.")
        pair = (review["user_id"], review["product_id"])
        if pair in seen_pairs:
            raise ValueError("Use only one review per user and product.")
        seen_pairs.add(pair)
    return products, reviews


def recommend(products, reviews, user_id=None, top_k=5):
    """Count positive reviews, rank products, then remove already-seen items.

    A rating of 4 or 5 is a positive interaction in this lesson.
    Ranking: positive count descending, average rating descending, ID ascending.
    No learned model or personalization is implemented yet.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    stats = {}
    seen_products = set()

    for review in reviews:
        product_id = review["product_id"]
        rating = review["rating"]

        if product_id not in stats:
            stats[product_id] = {"positive_count": 0, "total_rating": 0, "count": 0}

        stats[product_id]["total_rating"] += rating
        stats[product_id]["count"] += 1
        if rating >= 4:
            stats[product_id]["positive_count"] += 1

        # Exclude every reviewed item, even if the user disliked it.
        if review["user_id"] == user_id:
            seen_products.add(product_id)

    candidates = []
    for product_id, values in stats.items():
        if product_id in seen_products or values["positive_count"] == 0:
            continue
        candidates.append({
            "product_id": product_id,
            "name": products[product_id],
            "positive_count": values["positive_count"],
            "average_rating": values["total_rating"] / values["count"],
        })

    candidates.sort(key=lambda item: (
        -item["positive_count"], -item["average_rating"], item["product_id"]
    ))
    return candidates[:top_k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default=None, help="Try u1 or u2; omit for a new user.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    products, reviews = load_data(DATA_FILE)
    results = recommend(products, reviews, args.user, args.top_k)

    print("PRODUCT RECOMMENDER - Lesson 1")
    print("Data: FICTIONAL practice reviews; no real-world performance claim.")
    print(f"Loaded {len(products)} products and {len(reviews)} reviews.")
    print(f"Recommendations for: {args.user or 'a new user'}\n")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. {item['name']} | {item['positive_count']} positive reviews"
              f" | average {item['average_rating']:.2f}/5")
    if not results:
        print("No unseen products with positive reviews are available.")
    print("\nThis is a popularity baseline, not a personalized ML model yet.")


if __name__ == "__main__":
    main()

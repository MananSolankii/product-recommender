"""Lesson 5: item-item collaborative filtering from positive review overlap."""
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from evaluate import fit_popularity, split_by_time
from explore_data import DATA_FILE, prepare_reviews


class CollaborativeRecommender:
    def __init__(self, history):
        self.ids, self.seen = fit_popularity(history)
        if not self.ids:
            raise ValueError("Need at least one historical product.")
        self.index = {product: i for i, product in enumerate(self.ids)}
        positive = (history.loc[history["rating"] >= 4, ["user_id", "product_id"]]
                    .drop_duplicates())
        self.likes = positive.groupby("user_id")["product_id"].agg(set).to_dict()
        users = sorted(history["user_id"].unique())
        user_index = {user: i for i, user in enumerate(users)}
        rows = [user_index[u] for u in positive["user_id"]]
        columns = [self.index[p] for p in positive["product_id"]]
        # Rows = users, columns = products. 1 means observed positive feedback.
        interactions = csr_matrix(
            (np.ones(len(positive)), (rows, columns)),
            shape=(len(users), len(self.ids)),
        )
        # Transpose: now compare products using their vectors of user likes.
        self.similarity = cosine_similarity(
            interactions.T, dense_output=False,
        ).tocsr()
        # A product is not evidence for recommending itself.
        self.similarity.setdiag(0)
        self.similarity.eliminate_zeros()
        self.training_summary = {
            "historical_users": len(users),
            "historical_products": len(self.ids),
            "distinct_positive_user_product_pairs": len(positive),
            "users_with_two_or_more_positive_products": sum(
                len(items) >= 2 for items in self.likes.values()
            ),
            "product_pairs_sharing_positive_reviewers": int(self.similarity.nnz // 2),
        }

    def scores_for(self, user_id):
        liked = sorted(self.likes.get(user_id, set()))
        if not liked:
            return np.zeros(len(self.ids))
        indices = [self.index[product] for product in liked]
        # More similarity to the user's liked products gives a larger score.
        return np.asarray(self.similarity[indices].sum(axis=0)).ravel()

    def has_personal_signal(self, user_id):
        scores = self.scores_for(user_id)
        seen = self.seen.get(user_id, set())
        return any(scores[i] > 0 for i, product in enumerate(self.ids) if product not in seen)

    def recommend(self, user_id, k=5):
        if k < 1:
            raise ValueError("k must be positive.")
        scores = self.scores_for(user_id)
        excluded = self.seen.get(user_id, set())
        candidates = [i for i, product in enumerate(self.ids) if product not in excluded]
        # self.ids follows popularity order, so its index is our tie-break.
        candidates.sort(key=lambda i: (-float(scores[i]), i))
        return [self.ids[i] for i in candidates[:k]]


def demo():
    # Fictional teaching example: Alice likes A; Bob and Cara like both A and B.
    rows = [
        ("Alice", "A", 5), ("Bob", "A", 5), ("Bob", "B", 4),
        ("Cara", "A", 4), ("Cara", "B", 5), ("Dan", "C", 5),
    ]
    history = pd.DataFrame(rows, columns=["user_id", "product_id", "rating"])
    model = CollaborativeRecommender(history)
    print("FICTIONAL TEACHING EXAMPLE")
    print("Alice: A; Bob: A and B; Cara: A and B; Dan: C")
    a, b, c = (model.index[p] for p in ("A", "B", "C"))
    print(f"Similarity(A, B) = {model.similarity[a, b]:.3f}")
    print(f"Similarity(A, C) = {model.similarity[a, c]:.3f}")
    print("Recommendations for Alice:", model.recommend("Alice", k=2))
    print("B comes first because A and B share positive reviewers.")
    print("C fills the second slot via the popularity tie-break at score zero.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run the fictional teaching example.")
    parser.add_argument("--user", help="Historical user ID; unknown IDs get popularity.")
    args = parser.parse_args()
    if args.demo:
        demo()
        return
    if not DATA_FILE.exists():
        raise SystemExit("Run python download_data.py first.")
    reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
    train, _, _ = split_by_time(reviews)
    model = CollaborativeRecommender(train)
    # Choose an example from TRAINING history only, never by validation success.
    candidates = sorted(model.likes)
    user = args.user or next((u for u in candidates if model.has_personal_signal(u)), "new-user")
    print("Training summary:", model.training_summary)
    print("Example user:", user)
    print("Previously liked products:", sorted(model.likes.get(user, set())))
    print("Has unseen products with collaborative evidence:", model.has_personal_signal(user))
    scores = model.scores_for(user)
    for rank, product in enumerate(model.recommend(user), start=1):
        score = float(scores[model.index[product]])
        explanation = "positive-review overlap" if score > 0 else "popularity fallback"
        print(f"{rank}. {product} | score={score:.4f} | {explanation}")
    print("Scores are ranking signals, not probabilities. This demo is not an accuracy test.")


if __name__ == "__main__":
    main()

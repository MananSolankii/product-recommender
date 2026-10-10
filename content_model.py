"""TF-IDF recommendations from product descriptions and past positive reviews."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from evaluate import fit_popularity, split_by_time
from explore_data import DATA_FILE, prepare_reviews

METADATA_FILE = Path(__file__).resolve().parent / "data/raw/meta_Subscription_Boxes.jsonl.gz"


def load_metadata():
    if not METADATA_FILE.exists():
        raise SystemExit("Run python download_metadata.py first.")
    with gzip.open(METADATA_FILE, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    return {record["parent_asin"]: record for record in records}


def as_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(as_text(part) for part in value)
    return ""


def product_text(item):
    # Do not use review text, average_rating, rating_number, or bought_together.
    return " ".join(as_text(item.get(field)) for field in ("title", "features", "description"))


class ContentRecommender:
    def __init__(self, history, metadata):
        self.ranking, self.seen = fit_popularity(history)
        if not self.ranking:
            raise ValueError("Need at least one historical product.")
        self.metadata = metadata
        self.ids = self.ranking
        self.index = {product: i for i, product in enumerate(self.ids)}
        self.likes = (history.loc[history["rating"] >= 4]
                      .groupby("user_id")["product_id"].agg(set).to_dict())
        documents = [product_text(metadata.get(product, {})) for product in self.ids]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000,
        )
        # Vocabulary and IDF are fitted only on products present in history.
        try:
            self.matrix = self.vectorizer.fit_transform(documents)
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            self.matrix = csr_matrix((len(self.ids), 1))
            self.vectorizer = None

    def name(self, product):
        return as_text(self.metadata.get(product, {}).get("title")) or product

    def _scores(self, liked):
        rows = [self.index[product] for product in sorted(set(liked))
                if product in self.index and self.matrix[self.index[product]].nnz]
        if not rows:
            return None
        # Average the vectors of distinct liked products into one taste profile.
        profile = np.asarray(self.matrix[rows].mean(axis=0))
        return cosine_similarity(profile, self.matrix).ravel()

    def _rank(self, scores, excluded, k):
        if k < 1:
            raise ValueError("k must be positive.")
        indices = [i for i, product in enumerate(self.ids) if product not in excluded]
        if scores is not None:
            # self.ids is already popularity-sorted, so index breaks score ties.
            indices.sort(key=lambda i: (-float(scores[i]), i))
        return [self.ids[i] for i in indices[:k]]

    def recommend(self, user_id, k=5):
        scores = self._scores(self.likes.get(user_id, set()))
        # No usable positive history => popularity, still excluding all seen items.
        return self._rank(scores, self.seen.get(user_id, set()), k)

    def similar(self, product, k=5):
        if product not in self.index:
            raise ValueError("Choose a product in the historical catalog.")
        scores = self._scores({product})
        if scores is None:
            raise ValueError("This product has no usable text; choose another product.")
        return [(candidate, float(scores[self.index[candidate]]))
                for candidate in self._rank(scores, {product}, k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", help="List historical products whose names contain this text.")
    parser.add_argument("--item", help="Recommend products similar to this product ID.")
    args = parser.parse_args()
    if args.search and args.item:
        parser.error("Use --search or --item, not both.")
    if not DATA_FILE.exists():
        raise SystemExit("Run python download_data.py first.")
    reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
    train, _, _ = split_by_time(reviews)
    model = ContentRecommender(train, load_metadata())
    if args.search:
        matches = [p for p in model.ids if args.search.lower() in model.name(p).lower()]
        for product in matches[:20]:
            print(product, "|", model.name(product))
        if not matches:
            print("No matching historical products.")
        return
    coffee = [p for p in model.ids if "coffee" in model.name(p).lower()]
    seed = args.item or (coffee[0] if coffee else model.ids[0])
    try:
        recommendations = model.similar(seed)
    except ValueError as error:
        raise SystemExit(str(error))
    print("Selected interest:", seed, "|", model.name(seed))
    print("\nSIMILAR PRODUCTS")
    for rank, (product, score) in enumerate(recommendations, start=1):
        print(f"{rank}. {product} | {model.name(product)}")
        print(f"   Text similarity: {score:.3f}")
    print("\nSimilarity is not a probability that you will like the product.")
    print("Demo uses training-catalog products; product descriptions are a later snapshot.")


if __name__ == "__main__":
    main()

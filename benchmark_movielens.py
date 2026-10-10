"""Lesson 6: evaluate reusable recommenders on the MovieLens 100K benchmark."""
import hashlib
import inspect
import json
import platform
from pathlib import Path
import urllib.error
import urllib.request
from zipfile import BadZipFile, ZipFile

import pandas as pd
import sklearn
from collaborative_model import CollaborativeRecommender
from evaluate import evaluate, split_by_time

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data/raw/ml-100k.zip"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download():
    if ARCHIVE.exists():
        print("Using cached MovieLens archive.")
        return
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(".download")
    # Restart incomplete transfers. Never leave a partial file at the final path.
    for attempt in range(1, 4):
        print(f"Downloading MovieLens 100K (about 5 MB), attempt {attempt}/3...")
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "Recommender-Lesson6"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            with ZipFile(temporary) as archive:
                archive.getinfo("ml-100k/u.data")
                archive.getinfo("ml-100k/README")
                if archive.testzip() is not None:
                    raise ValueError("Archive checksum verification failed.")
            temporary.replace(ARCHIVE)
            print(f"Saved: {ARCHIVE}")
            return
        except (OSError, ValueError, EOFError, BadZipFile, KeyError,
                urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise SystemExit(
                    f"Download failed: {error}\n"
                    "Download ml-100k.zip from https://grouplens.org/datasets/movielens/100k/ "
                    "and put it in data/raw/; then run this command again."
                )


def load_ratings(archive_path):
    # Read only ratings. Do not extract archive paths or use demographic fields.
    with ZipFile(archive_path) as archive:
        with archive.open("ml-100k/u.data") as source:
            ratings = pd.read_csv(
                source, sep="\t", header=None,
                names=["user_id", "product_id", "rating", "timestamp"],
            )
    if ratings.empty or ratings.isna().any().any():
        raise ValueError("Ratings are empty or contain missing values.")
    for column in ratings.columns:
        ratings[column] = pd.to_numeric(ratings[column], errors="raise")
    if not ratings["rating"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Ratings must be whole stars from 1 to 5.")
    for column in ["user_id", "product_id"]:
        if not ((ratings[column] > 0) & (ratings[column] % 1 == 0)).all():
            raise ValueError("MovieLens identifiers must be positive integers.")
        ratings[column] = ratings[column].astype("int64").astype(str)
    if ratings.duplicated(["user_id", "product_id"]).any():
        raise ValueError("Unexpected duplicate user/movie pairs in the benchmark.")
    # MovieLens uses SECONDS, unlike the milliseconds in Amazon Reviews 2023.
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True, errors="raise")
    return ratings.sort_values("timestamp").reset_index(drop=True)


def main():
    if "recommend_fn" not in inspect.signature(evaluate).parameters:
        raise SystemExit("Update evaluate.py from the project ZIP; recommend_fn support is required.")
    download()
    ratings = load_ratings(ARCHIVE)
    train, validation, test = split_by_time(ratings)
    model = CollaborativeRecommender(train)
    popularity = evaluate(train, validation)
    collaborative = evaluate(train, validation, recommend_fn=model.recommend)
    positive_counts = train.loc[train["rating"] >= 4].groupby("user_id")["product_id"].nunique()
    summary = {
        "ratings": len(ratings),
        "users": int(ratings["user_id"].nunique()),
        "movies": int(ratings["product_id"].nunique()),
        "minimum_ratings_per_user_in_full_dataset": int(ratings.groupby("user_id").size().min()),
        "training_users_with_two_or_more_positive_movies": int((positive_counts >= 2).sum()),
        "median_distinct_movies_per_training_user": float(
            train.groupby("user_id")["product_id"].nunique().median()
        ),
    }
    split = {
        name: {"rows": len(part), "start": str(part.timestamp.min()), "end": str(part.timestamp.max())}
        for name, part in (("train", train), ("validation", validation), ("test", test))
    }
    report = {
        "dataset": "MovieLens 100K",
        "source": "https://grouplens.org/datasets/movielens/100k/",
        "source_sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        "stage": "validation",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "summary": summary,
        "split": split,
        "popularity": popularity,
        "collaborative": collaborative,
        "scope": (
            "Selected population: source already requires at least 20 ratings per user. "
            "Global chronological 70/15/15 split, not the supplied official folds. "
            "Results concern movie ratings, not Amazon products."
        ),
    }
    print("\nMOVIELENS 100K: DATA SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nVALIDATION RESULTS: K=5")
    print(f"{'Model':<15} {'Group':<18} {'Users':>6} {'Recall@5':>10} {'NDCG@5':>10}")
    for name, result in (("popularity", popularity), ("collaborative", collaborative)):
        for group in ("all_users", "returning_users", "new_users"):
            row = result[group]
            recall = "n/a" if row["recall_at_5"] is None else f"{row['recall_at_5']:.4f}"
            ndcg = "n/a" if row["ndcg_at_5"] is None else f"{row['ndcg_at_5']:.4f}"
            print(f"{name:<15} {group:<18} {row['evaluated_users']:>6} {recall:>10} {ndcg:>10}")
    print("\n" + report["scope"])
    print("Compare models WITHIN this dataset. Scores across datasets are not directly comparable.")
    print("Test scores are reserved; this run uses no test ratings to fit or score a model.")
    destination = ROOT / "artifacts/lesson6_movielens_validation.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

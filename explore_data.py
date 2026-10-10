"""Lesson 2: inspect real reviews with pandas. No model training yet."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "raw" / "Subscription_Boxes.jsonl.gz"


def prepare_reviews(raw):
    """Select four fields; report bad values and remove unusable rows."""
    columns = ["user_id", "parent_asin", "rating", "timestamp"]
    missing_columns = set(columns) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    df = raw[columns].rename(columns={"parent_asin": "product_id"}).copy()
    for column in ["user_id", "product_id"]:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    # Source timestamps are milliseconds since 1970-01-01.
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms", utc=True, errors="coerce",
    )
    missing = {key: int(value) for key, value in df.isna().sum().items()}
    invalid_ratings = df["rating"].notna() & ~df["rating"].isin([1, 2, 3, 4, 5])
    invalid_dates = df["timestamp"].notna() & (
        (df["timestamp"] < pd.Timestamp("1996-01-01", tz="UTC")) |
        (df["timestamp"] >= pd.Timestamp("2024-01-01", tz="UTC"))
    )
    usable = df.notna().all(axis=1) & ~invalid_ratings & ~invalid_dates
    clean = df.loc[usable].copy()
    duplicates = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    audit = {
        "raw_rows": len(raw),
        "missing_or_unparseable": missing,
        "out_of_range_ratings": int(invalid_ratings.sum()),
        "out_of_range_dates": int(invalid_dates.sum()),
        "unusable_rows_removed": int((~usable).sum()),
        "duplicate_events_removed": duplicates,
        "clean_rows": len(clean),
    }
    return clean, audit


def summarize(df):
    user_counts = df.groupby("user_id").size()
    unique_items_per_user = df.groupby("user_id")["product_id"].nunique()
    # Multiple reviews of the same product do not provide multiple item preferences.
    positive_items = df.loc[df["rating"] >= 4].groupby("user_id")["product_id"].nunique()
    return {
        "users": int(df["user_id"].nunique()),
        "products": int(df["product_id"].nunique()),
        "date_start": str(df["timestamp"].min()),
        "date_end": str(df["timestamp"].max()),
        "users_with_one_review": int((user_counts == 1).sum()),
        "users_with_at_least_two_distinct_products": int((unique_items_per_user >= 2).sum()),
        "users_with_at_least_three_distinct_products": int((unique_items_per_user >= 3).sum()),
        "users_with_at_least_three_positive_distinct_products": int((positive_items >= 3).sum()),
        "repeated_user_product_rows": int(df.duplicated(["user_id", "product_id"]).sum()),
        "rating_counts": {
            str(rating): int((df["rating"] == rating).sum()) for rating in range(1, 6)
        },
        "positive_review_percentage": round(float((df["rating"] >= 4).mean() * 100), 2),
    }


def main():
    if not DATA_FILE.exists():
        raise SystemExit("Dataset missing. Run: python download_data.py")
    # JSONL has one review per line. pandas also decompresses .gz automatically.
    raw = pd.read_json(DATA_FILE, lines=True, convert_dates=False)
    reviews, audit = prepare_reviews(raw)
    if reviews.empty:
        raise SystemExit("No usable rows remain. Check the dataset and cleaning report.")
    summary = summarize(reviews)

    print("LESSON 2: REAL REVIEW DATA\n")
    print("FIRST FIVE CLEAN ROWS")
    print(reviews.head().to_string(index=False))
    print("\nCLEANING REPORT")
    print(json.dumps(audit, indent=2))
    print("\nDATA SUMMARY")
    print(json.dumps(summary, indent=2))

    # Describe positive popularity. This is not a trained or evaluated recommender.
    positive = reviews.loc[reviews["rating"] >= 4]
    top = (positive.groupby("product_id")["user_id"].nunique()
           .rename("positive_users").reset_index()
           .sort_values(["positive_users", "product_id"], ascending=[False, True])
           .head(10))
    print("\nTOP PRODUCTS BY DISTINCT POSITIVE REVIEWERS")
    print(top.to_string(index=False))
    print("\nProduct IDs are shown because we have not loaded product metadata yet.")
    print("These are exploration statistics, not model accuracy scores.")

    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    report = {"source": "Amazon Reviews 2023 / Subscription_Boxes", "audit": audit, "summary": summary}
    (output / "lesson2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Preserve dates as ISO strings for the next lesson; raw data remains untouched.
    reviews.to_json(output / "clean_reviews.jsonl", orient="records", lines=True, date_format="iso")
    print(f"\nSaved the summary and cleaned reviews in {output}")


if __name__ == "__main__":
    main()

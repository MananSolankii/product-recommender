"""Download the small, real dataset used in Lesson 2."""
import gzip
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "data" / "raw" / "Subscription_Boxes.jsonl.gz"
URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/review_categories/Subscription_Boxes.jsonl.gz"
)


def download():
    if DESTINATION.exists():
        print(f"Already downloaded: {DESTINATION}")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".download")
    print("Downloading Amazon Reviews 2023: Subscription Boxes...")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "ProductRecommender-Lesson2"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        # Check the response is a complete gzip containing JSON review records.
        count = 0
        with gzip.open(temporary, "rt", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if not {"user_id", "parent_asin", "rating", "timestamp"} <= record.keys():
                    raise ValueError("Unexpected dataset fields.")
                count += 1
        if count == 0:
            raise ValueError("Downloaded dataset is empty.")
        temporary.replace(DESTINATION)
        print(f"Saved {count:,} reviews to {DESTINATION}")
    except (OSError, ValueError, EOFError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed: {error}\n"
            "Retry later or download the Subscription_Boxes review file from "
            "https://amazon-reviews-2023.github.io/ and place it in data/raw/."
        )


if __name__ == "__main__":
    download()

"""Download product titles/descriptions for Lesson 4."""
import gzip
import json
from pathlib import Path
import urllib.error
import urllib.request

DESTINATION = Path(__file__).resolve().parent / "data/raw/meta_Subscription_Boxes.jsonl.gz"
URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
       "raw/meta_categories/meta_Subscription_Boxes.jsonl.gz")


def main():
    if DESTINATION.exists():
        print(f"Already downloaded: {DESTINATION}")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".download")
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "ProductRecommender-Lesson4"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        count = 0
        with gzip.open(temporary, "rt", encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                if not isinstance(item, dict) or not item.get("parent_asin"):
                    raise ValueError("Unexpected metadata format.")
                count += 1
        if count == 0:
            raise ValueError("Empty metadata download.")
        temporary.replace(DESTINATION)
        print(f"Downloaded metadata for {count:,} products.")
    except (OSError, ValueError, EOFError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {error}. Retry or use the metadata link at "
                         "https://amazon-reviews-2023.github.io/.")


if __name__ == "__main__":
    main()

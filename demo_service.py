"""Step 9: read-only, session-interest adapter for the fixed TF-IDF model."""
from numbers import Integral
import numpy as np
import pandas as pd

from content_model import ContentRecommender, METADATA_FILE, load_metadata
from evaluate import split_by_time
from explore_data import DATA_FILE, prepare_reviews
from final_evaluate import ROOT, load_fixed_selection, verify_inputs


class DemoService:
    """Never mutate the shared fitted model with a visitor's interests."""

    def __init__(self, model, history_end):
        self.model = model
        self.history_end = str(history_end)
        self.catalog = tuple(model.ids)
        self.terms = (model.vectorizer.get_feature_names_out()
                      if model.vectorizer is not None else np.array([]))

    def label(self, product):
        return f"{self.model.name(product)[:130]} [{product}]"

    def recommendations(self, liked, excluded=(), k=5):
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= 10:
            raise ValueError("Choose between 1 and 10 results.")
        liked, excluded = set(liked), set(excluded)
        if len(liked) > 10 or len(excluded) > 10:
            raise ValueError("Select at most 10 interests and 10 excluded items.")
        if (liked | excluded) - set(self.catalog):
            raise ValueError("Choose items from the displayed catalog.")
        if liked & excluded:
            raise ValueError("An item cannot be both liked and excluded. Remove it from one list.")
        # Reuse the evaluated scoring/ranking rules without changing any model settings.
        scores = self.model._scores(liked)
        ranked = self.model._rank(scores, liked | excluded, k)
        usable = [self.model.index[p] for p in sorted(liked)
                  if self.model.matrix[self.model.index[p]].nnz]
        profile = np.asarray(self.model.matrix[usable].mean(axis=0)).ravel() if usable else None
        rows = []
        for product in ranked:
            i = self.model.index[product]
            score = float(scores[i]) if scores is not None else 0.0
            shared_terms = []
            if score > 0 and profile is not None and self.terms.size:
                # Largest positive contributions to the profile/item dot product.
                weights = self.model.matrix[i].toarray().ravel() * profile
                indices = np.flatnonzero(weights > 0)
                top = sorted(indices, key=lambda j: (-weights[j], str(self.terms[j])))[:3]
                shared_terms = [str(self.terms[j]) for j in top]
            rows.append({
                "product_id": product,
                "title": self.model.name(product),
                "score": score,
                "reason": "Text match" if score > 0 else "Popularity fallback",
                "shared_terms": shared_terms,
            })
        return rows


def file_signature():
    """Invalidate the app cache when input or selection files change."""
    files = (DATA_FILE, METADATA_FILE, ROOT / "artifacts/lesson7_products_selection.json")
    result = []
    for filename in files:
        if not filename.exists():
            raise ValueError(
                f"Missing {filename.name}. Keep the data/raw folder and Step 7 product "
                "selection report in this project. See LESSON_9.md."
            )
        stat = filename.stat()
        result.append((filename.name, stat.st_size, stat.st_mtime_ns))
    return tuple(result)


def build_demo():
    """Fit on train + validation only. Never call the final evaluator."""
    try:
        _, selection = load_fixed_selection("products")
        verify_inputs(selection, [DATA_FILE, METADATA_FILE])
        metadata = load_metadata()
    except SystemExit as error:
        raise ValueError(str(error)) from error
    reviews, _ = prepare_reviews(pd.read_json(DATA_FILE, lines=True, convert_dates=False))
    train, validation, _ = split_by_time(reviews)
    history = pd.concat([train, validation], ignore_index=True)
    return DemoService(ContentRecommender(history, metadata), history.timestamp.max())

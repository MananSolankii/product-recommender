import unittest
import pandas as pd
from content_model import ContentRecommender
from evaluate import evaluate, fit_popularity
from test_evaluation import frame


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.history = frame([("u", "coffee1", 5, 1), ("v", "coffee2", 4, 2),
                              ("w", "yoga", 5, 3), ("u", "disliked", 1, 4)])
        self.metadata = {
            "coffee1": {"title": "coffee beans roast"},
            "coffee2": {"title": "coffee beans medium roast"},
            "yoga": {"title": "yoga mat stretching"},
            "disliked": {"title": "coffee beans"},
            "future-only": {"title": "forbiddenfutureword"},
        }
        self.model = ContentRecommender(self.history, self.metadata)

    def test_preferences_and_seen_items(self):
        items = self.model.recommend("u", 5)
        self.assertEqual(items[0], "coffee2")
        self.assertNotIn("coffee1", items)
        self.assertNotIn("disliked", items)
        self.assertNotIn("future-only", items)
        self.assertNotIn("forbiddenfutureword", self.model.vectorizer.vocabulary_)

    def test_new_user_and_missing_text_fallback(self):
        ranking, _ = fit_popularity(self.history)
        self.assertEqual(self.model.recommend("new"), ranking)
        empty = ContentRecommender(self.history, {})
        self.assertEqual(empty.recommend("u"), [p for p in ranking if p not in {"coffee1", "disliked"}])

    def test_duplicate_likes_and_seed_exclusion(self):
        doubled = pd.concat([self.history, self.history.iloc[[0]]], ignore_index=True)
        model2 = ContentRecommender(doubled, self.metadata)
        self.assertEqual(self.model.recommend("u"), model2.recommend("u"))
        self.assertNotIn("coffee1", [p for p, _ in self.model.similar("coffee1")])
        with self.assertRaises(ValueError):
            self.model.similar("future-only")

    def test_shared_evaluation_and_invalid_predictions(self):
        future = frame([("u", "coffee2", 5, 5), ("new", "coffee2", 5, 6)])
        result = evaluate(self.history, future, recommend_fn=self.model.recommend)
        self.assertEqual(result["returning_users"]["recall_at_5"], 1)
        with self.assertRaises(ValueError):
            evaluate(self.history, future, recommend_fn=lambda u, k: ["future-only"])
        with self.assertRaises(ValueError):
            evaluate(self.history, future, recommend_fn=lambda u, k: ["coffee1"])


if __name__ == "__main__":
    unittest.main()

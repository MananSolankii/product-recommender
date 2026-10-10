import unittest
import numpy as np
import pandas as pd
from collaborative_model import CollaborativeRecommender
from evaluate import evaluate, fit_popularity
from test_evaluation import frame


class CollaborativeTests(unittest.TestCase):
    def setUp(self):
        self.history = frame([
            ("Alice", "A", 5, 1), ("Bob", "A", 5, 2), ("Bob", "B", 4, 3),
            ("Cara", "A", 4, 4), ("Cara", "B", 5, 5), ("Dan", "C", 5, 6),
        ])
        self.model = CollaborativeRecommender(self.history)

    def test_hand_computed_similarity(self):
        a, b, c = (self.model.index[p] for p in ("A", "B", "C"))
        self.assertAlmostEqual(self.model.similarity[a, b], 2 / np.sqrt(3 * 2))
        self.assertEqual(self.model.similarity[a, c], 0)
        self.assertEqual(self.model.similarity[a, a], 0)
        self.assertEqual(self.model.recommend("Alice", 1), ["B"])
        self.assertNotIn("A", self.model.recommend("Alice"))

    def test_duplicate_reviews_and_fallback(self):
        doubled = pd.concat([self.history, self.history.iloc[[1, 2]]], ignore_index=True)
        other = CollaborativeRecommender(doubled)
        np.testing.assert_allclose(other.similarity.toarray(), self.model.similarity.toarray())
        self.assertEqual(self.model.recommend("unknown"), fit_popularity(self.history)[0])
        self.assertFalse(self.model.has_personal_signal("unknown"))

    def test_dislikes_and_no_positive_data(self):
        history = pd.concat([self.history, frame([("Alice", "B", 1, 7)])], ignore_index=True)
        self.assertEqual(CollaborativeRecommender(history).recommend("Alice"), ["C"])
        negative = frame([("u", "a", 1, 1), ("v", "b", 2, 2)])
        model = CollaborativeRecommender(negative)
        self.assertEqual(model.recommend("u"), ["b"])
        self.assertFalse(model.has_personal_signal("u"))

    def test_future_items_and_shared_evaluation(self):
        future = frame([("Alice", "B", 5, 8), ("Alice", "future-only", 5, 9)])
        result = evaluate(self.history, future, recommend_fn=self.model.recommend)
        self.assertEqual(result["returning_users"]["recall_at_5"], 0.5)
        self.assertEqual(result["all_users"]["targets_absent_from_history_catalog"], 1)
        self.assertNotIn("future-only", self.model.ids)


if __name__ == "__main__":
    unittest.main()

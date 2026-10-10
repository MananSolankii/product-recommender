"""Check pruning, ranking safety, and the fixed model-selection rule."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from collaborative_model import CollaborativeRecommender
from evaluate import evaluate, fit_popularity
from neighbor_model import NeighborRecommender, keep_top_neighbors
from test_evaluation import frame
from tune_models import select_best


class NeighborTests(unittest.TestCase):
    def test_hand_calculated_pruning_and_ties(self):
        original = csr_matrix([
            [0, .9, .4, .4],
            [.9, 0, .2, 0],
            [.4, .2, 0, 0],
            [.4, 0, 0, 0],
        ])
        result = keep_top_neighbors(original, 2)
        expected = original.toarray()
        expected[0, 3] = 0  # Equal .4 ties favor the lower popularity index.
        np.testing.assert_allclose(result.toarray(), expected)
        self.assertEqual(original[0, 3], .4)  # Input was not modified.
        self.assertEqual(result[3, 0], .4)  # Pruned links may be asymmetric.
        self.assertTrue(np.all(np.diff(result.indptr) <= 2))

    def test_large_limit_matches_base_and_safe_fallback(self):
        history = frame([
            ("Alice", "A", 5, 1), ("Bob", "A", 5, 2), ("Bob", "B", 5, 3),
            ("Cara", "A", 5, 4), ("Cara", "B", 5, 5), ("Dan", "C", 5, 6),
            ("Alice", "C", 1, 7),
        ])
        base = CollaborativeRecommender(history)
        limited = NeighborRecommender(history, neighbors=100)
        np.testing.assert_allclose(limited.similarity.toarray(), base.similarity.toarray())
        self.assertEqual(limited.recommend("Alice"), ["B"])
        self.assertEqual(limited.recommend("unknown"), fit_popularity(history)[0])
        future = frame([("Alice", "B", 5, 8), ("Alice", "future-only", 5, 9)])
        result = evaluate(history, future, recommend_fn=limited.recommend)
        self.assertEqual(result["all_users"]["recall_at_5"], .5)
        self.assertNotIn("future-only", limited.ids)

    def test_empty_positive_history_and_invalid_settings(self):
        history = frame([("u", "a", 1, 1), ("v", "b", 2, 2)])
        model = NeighborRecommender(history, neighbors=1)
        self.assertEqual(model.similarity.nnz, 0)
        self.assertEqual(model.recommend("u"), ["b"])
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                NeighborRecommender(history, neighbors=invalid)

    def test_selection_metric_priority_and_exact_ties(self):
        def candidate(name, ndcg, recall):
            return {"config": {"name": name},
                    "metrics": {"all_users": {"ndcg_at_5": ndcg, "recall_at_5": recall}}}
        a = candidate("popularity", .2, .1)
        b = candidate("neighbors_10", .1, .9)
        self.assertIs(select_best([a, b]), a)  # NDCG takes priority over Recall.
        c = candidate("content", .2, .2)
        self.assertIs(select_best([a, c]), c)
        self.assertIs(select_best([a, candidate("same", .2, .1)]), a)


if __name__ == "__main__":
    unittest.main()

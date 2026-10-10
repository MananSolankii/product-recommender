import unittest
import numpy as np
from content_model import ContentRecommender
from demo_service import DemoService
from test_evaluation import frame


def fixture_service():
    history = frame([
        ("u", "a", 5, 1), ("v", "a", 5, 2), ("v", "b", 5, 3),
        ("w", "c", 5, 4), ("z", "d", 1, 5), ("u", "d", 1, 6),
    ])
    metadata = {
        "a": {"title": "Coffee beans box"},
        "b": {"title": "Coffee beans subscription"},
        "c": {"title": "Herbal tea collection"},
        "d": {},
    }
    return DemoService(ContentRecommender(history, metadata), history.timestamp.max())


class DemoTests(unittest.TestCase):
    def test_interest_ranking_matches_evaluated_model(self):
        service = fixture_service()
        rows = service.recommendations(["a"], ["d"])
        self.assertEqual([r["product_id"] for r in rows], service.model.recommend("u"))
        self.assertEqual(rows[0]["product_id"], "b")
        self.assertGreater(rows[0]["score"], 0)
        self.assertIn("coffee", rows[0]["shared_terms"])
        self.assertEqual(rows[0]["reason"], "Text match")

    def test_popularity_and_missing_text_fallback(self):
        service = fixture_service()
        rows = service.recommendations([])
        self.assertEqual([r["product_id"] for r in rows], list(service.catalog))
        self.assertTrue(all(r["reason"] == "Popularity fallback" for r in rows))
        rows = service.recommendations(["d"])
        self.assertNotIn("d", [r["product_id"] for r in rows])
        self.assertTrue(all(r["score"] == 0 for r in rows))

    def test_request_does_not_mutate_shared_model(self):
        service = fixture_service()
        matrix = service.model.matrix.toarray().copy()
        before = service.recommendations(["a"])
        service.recommendations(["c"], ["a"])
        self.assertEqual(service.recommendations(["a"]), before)
        np.testing.assert_array_equal(service.model.matrix.toarray(), matrix)
        self.assertEqual(service.model.likes["u"], {"a"})
        self.assertEqual(service.model.seen["u"], {"a", "d"})

    def test_input_guards_and_empty_results(self):
        service = fixture_service()
        for liked, excluded in ((["unknown"], []), (["a"], ["a"])):
            with self.assertRaises(ValueError):
                service.recommendations(liked, excluded)
        for k in (0, 11, 1.5, True):
            with self.assertRaises(ValueError):
                service.recommendations([], k=k)
        self.assertEqual(service.recommendations([], service.catalog), [])
        self.assertEqual(service.recommendations(["a", "a"]), service.recommendations(["a"]))


if __name__ == "__main__":
    unittest.main()

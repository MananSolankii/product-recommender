"""Small checks for metric correctness and time leakage. Run with unittest."""
import unittest
from math import log2
import pandas as pd
from evaluate import evaluate, fit_popularity, ranking_metrics, split_by_time


def frame(rows):
    df = pd.DataFrame(rows, columns=["user_id", "product_id", "rating", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


class EvaluationTests(unittest.TestCase):
    def test_known_metric_values(self):
        self.assertEqual(ranking_metrics(["a", "b"], {"a", "b"}, 5), (1, 1))
        self.assertEqual(ranking_metrics(["x"], {"a"}, 5), (0, 0))
        recall, ndcg = ranking_metrics(["x", "a"], {"a"}, 5)
        self.assertEqual(recall, 1)
        self.assertAlmostEqual(ndcg, 1 / log2(3))
        recall, _ = ranking_metrics(["a"], {"a", "b"}, 5)
        self.assertEqual(recall, 0.5)

    def test_time_order_and_tied_timestamps(self):
        df = frame([("u", str(i), 5, i // 2) for i in range(20)])
        train, val, test = split_by_time(df.sample(frac=1, random_state=1))
        self.assertEqual(len(train) + len(val) + len(test), 20)
        self.assertLess(train.timestamp.max(), val.timestamp.min())
        self.assertLess(val.timestamp.max(), test.timestamp.min())

    def test_seen_dislikes_unknown_targets_and_new_users(self):
        history = frame([("u", "a", 1, 1), ("v", "b", 5, 2), ("v", "c", 5, 3)])
        future = frame([("u", "a", 5, 4), ("u", "b", 5, 5),
                        ("u", "new", 5, 6), ("new-user", "b", 5, 7),
                        ("only-dislikes", "b", 1, 8)])
        scores = evaluate(history, future)
        self.assertEqual(scores["returning_users"]["recall_at_5"], 0.5)
        self.assertEqual(scores["new_users"]["recall_at_5"], 1)
        self.assertEqual(scores["all_users"]["recall_at_5"], 0.75)
        self.assertEqual(scores["all_users"]["targets_absent_from_history_catalog"], 1)
        self.assertEqual(scores["users_without_new_positive_targets_excluded"], 1)
        self.assertNotIn("new", fit_popularity(history)[0])
        with self.assertRaises(ValueError):
            evaluate(future, history)

    def test_repeated_votes_do_not_inflate_popularity(self):
        history = frame([("u", "a", 5, 1), ("u", "a", 5, 2),
                         ("u", "b", 5, 3), ("v", "b", 4, 4)])
        self.assertEqual(fit_popularity(history)[0][0], "b")


if __name__ == "__main__":
    unittest.main()

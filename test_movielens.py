import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from benchmark_movielens import load_ratings


class MovieLensLoaderTests(unittest.TestCase):
    def load_text(self, text):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "fixture.zip"
            with ZipFile(archive, "w") as output:
                output.writestr("ml-100k/u.data", text)
            return load_ratings(archive)

    def test_seconds_ids_and_sorting(self):
        data = self.load_text("1\t2\t5\t874965760\n2\t3\t4\t874965758\n")
        self.assertEqual(len(data), 2)
        self.assertTrue(data.timestamp.dt.year.eq(1997).all())
        self.assertTrue(data.timestamp.is_monotonic_increasing)
        self.assertEqual(data.iloc[0]["user_id"], "2")
        self.assertEqual(data.iloc[0]["product_id"], "3")

    def test_invalid_and_missing_ratings(self):
        for text in ("1\t2\t9\t874965758\n", "1\t2\t\t874965758\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load_text(text)

    def test_duplicate_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.load_text("1\t2\t5\t874965758\n1\t2\t4\t874965760\n")


if __name__ == "__main__":
    unittest.main()

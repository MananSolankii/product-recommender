import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import final_evaluate as final


class FinalEvaluationTests(unittest.TestCase):
    def write_selection(self, directory, dataset, config=None, stage="validation_selection"):
        report = {
            "stage": stage,
            "dataset": dataset,
            "selected_config": config or final.EXPECTED_CONFIGS[dataset],
            "input_sha256": {},
        }
        output = Path(directory) / "artifacts"
        output.mkdir(exist_ok=True)
        path = output / f"lesson7_{dataset}_selection.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_accepts_only_fixed_step7_selection(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(final, "ROOT", Path(temp)):
            path = self.write_selection(temp, "movielens")
            returned, report = final.load_fixed_selection("movielens")
            self.assertEqual(returned, path)
            self.assertEqual(report["selected_config"], final.EXPECTED_CONFIGS["movielens"])

    def test_rejects_changed_config_or_wrong_stage(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(final, "ROOT", Path(temp)):
            self.write_selection(temp, "products", {"name": "popularity", "kind": "popularity"})
            with self.assertRaises(SystemExit):
                final.load_fixed_selection("products")
            self.write_selection(temp, "products", stage="final_held_out_test")
            with self.assertRaises(SystemExit):
                final.load_fixed_selection("products")

    def test_hash_verification_blocks_mixed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data.txt"
            data.write_text("first", encoding="utf-8")
            selection = {"input_sha256": {"data.txt": final.sha256(data)}}
            self.assertEqual(final.verify_inputs(selection, [data]), selection["input_sha256"])
            data.write_text("changed", encoding="utf-8")
            with self.assertRaises(SystemExit):
                final.verify_inputs(selection, [data])


if __name__ == "__main__":
    unittest.main()

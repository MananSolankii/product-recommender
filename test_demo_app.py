"""UI behavior checks with fictional data; no browser, downloads or test scoring."""
from pathlib import Path
import unittest
from unittest.mock import patch
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_demo import fixture_service

APP = Path(__file__).resolve().parent / "app.py"


class DemoAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def test_start_select_submit_and_reset(self):
        with patch("demo_service.build_demo", return_value=fixture_service()), \
             patch("demo_service.file_signature", return_value=("fixture",)):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.title[0].value, "Product Compass")
            app.multiselect(key="liked").select("a")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["submitted"], (("a",), ()))
            self.assertTrue(any("Coffee beans subscription" in t.value for t in app.text))
            app.button[1].click().run()
            self.assertEqual(app.multiselect(key="liked").value, [])
            self.assertEqual(len(app.exception), 0)

    def test_conflicting_choices_show_actionable_error(self):
        with patch("demo_service.build_demo", return_value=fixture_service()), \
             patch("demo_service.file_signature", return_value=("fixture",)):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            app.multiselect(key="liked").select("a")
            app.multiselect(key="excluded").select("a")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("both liked and excluded", app.error[0].value)

    def test_missing_data_shows_guidance_without_crash(self):
        with patch("demo_service.file_signature", side_effect=ValueError("Missing data")):
            app = AppTest.from_file(str(APP), default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.error[0].value, "Missing data")


if __name__ == "__main__":
    unittest.main()

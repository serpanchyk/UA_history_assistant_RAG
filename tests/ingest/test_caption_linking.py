"""Tests for caption linking helpers: ensure texts on the same page are found, nearest text is chosen, and images get linked to captions."""

import unittest
import pandas as pd

from src.ingest.caption_linking import find_texts_on_same_page, find_closest_text, link_image_to_text
from tests.fixtures.mock_classes import MockImageRow

class TestGetTextsForImage(unittest.TestCase):
    """Ensure texts on the same document page are found and associated with an image row."""

    def test_texts_found(self):
        """Assert that texts on the same doc/page are returned for the image row."""
        df_texts = pd.DataFrame({
            "doc_id": [0, 0, 1],
            "page": [0, 1, 0],
            "bbox": [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            "text": ['A', 'B', 'C']
        })

        groups = df_texts.groupby(['doc_id', 'page'])

        image_row = MockImageRow(
            path='img.png', bbox=[0, 0, 0, 0], doc_id=0, page=0
        )

        results = find_texts_on_same_page(image_row, groups)

        self.assertEqual(results.at[0, 'text'], 'A')
        self.assertEqual(len(results), 1)

    def texts_not_found(self):
        """Ensure None is returned when no matching page/doc_id group exists for the image."""
        df_texts = pd.DataFrame({
            "doc_id": [0, 1],
            "page": [1, 2],
            "bbox": [(0, 0, 0, 0), (0, 0, 0, 0)],
            "text": ['A', 'B']
        })

        groups = df_texts.groupby(['doc_id', 'page'])

        image_row = MockImageRow(
            path='img.png', bbox=[0, 0, 0, 0], doc_id=0, page=2
        )

        results = find_texts_on_same_page(image_row, groups)

        self.assertIsNone(results)

class TestFindClosestText(unittest.TestCase):
    """Verify that the closest text (by bbox centroid) to a given image bbox is selected."""

    def test_finds_nearest(self):
        """Confirm nearest text (centroid distance) is chosen from multiple candidates."""
        texts_df = pd.DataFrame({
            "bbox": [
                (0, 0, 5, 5),
                (10, 10, 20, 20),
            ],
            "text": ["near", "far"],
        })

        image_bbox = (1, 1, 3, 3)

        self.assertEqual(
            find_closest_text(image_bbox, texts_df),
            "near"
        )

    def test_empty_texts(self):
        """Return None when the texts DataFrame is empty."""
        texts_df = pd.DataFrame(columns=["bbox", "text"])

        self.assertIsNone(
            find_closest_text((0, 0, 1, 1), texts_df)
        )

class TestLinkImageToText(unittest.TestCase):
    """Check that images are linked to their correct captions when texts and images are provided."""

    def test_happy_path(self):
        """Verify that the function attaches the closest caption column to images in the same page."""
        df_images = pd.DataFrame({
            "doc_id": [1],
            "page": [1],
            "bbox": [(0, 0, 4, 4)],
        })

        df_texts = pd.DataFrame({
            "doc_id": [1, 1],
            "page": [1, 1],
            "bbox": [(0, 0, 5, 5), (20, 20, 30, 30)],
            "text": ["caption", "other"],
        })

        result = link_image_to_text(df_images, df_texts)

        self.assertIn("caption", result.columns)
        self.assertEqual(result.loc[0, "caption"], "caption")


if __name__ == '__main__':
    unittest.main()
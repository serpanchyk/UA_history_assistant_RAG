import unittest
import pandas as pd

from src.ingest.caption_linking import get_texts_for_image, find_closest_text, link_image_to_text
from tests.fixtures.mock_classes import MocImageRow

class TestGetTextsForImage(unittest.TestCase):
    def test_texts_found(self):
        df_texts = pd.DataFrame({
            "doc_id": [0, 0, 1],
            "page": [0, 1, 0],
            "bbox": [(0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)],
            "text": ['A', 'B', 'C']
        })

        groups = df_texts.groupby(['doc_id', 'page'])

        image_row = MocImageRow(
            path='img.png', bbox=[0, 0, 0, 0], doc_id=0, page=0
        )

        results = get_texts_for_image(image_row, groups)

        self.assertEqual(results.at[0, 'text'], 'A')
        self.assertEqual(len(results), 1)

    def texts_not_found(self):
        df_texts = pd.DataFrame({
            "doc_id": [0, 1],
            "page": [1, 2],
            "bbox": [(0, 0, 0, 0), (0, 0, 0, 0)],
            "text": ['A', 'B']
        })

        groups = df_texts.groupby(['doc_id', 'page'])

        image_row = MocImageRow(
            path='img.png', bbox=[0, 0, 0, 0], doc_id=0, page=2
        )

        results = get_texts_for_image(image_row, groups)

        self.assertIsNone(results)

class TestFindClosestText(unittest.TestCase):

    def test_finds_nearest(self):
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
        texts_df = pd.DataFrame(columns=["bbox", "text"])

        self.assertIsNone(
            find_closest_text((0, 0, 1, 1), texts_df)
        )

class TestLinkImageToText(unittest.TestCase):

    def test_happy_path(self):
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
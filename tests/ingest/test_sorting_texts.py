import unittest

import pandas as pd

from src.ingest.sorting_texts import bbox_sort, sort_texts


class TestSortingTexts(unittest.TestCase):

    def bbox_sort_y_by_desc_x_by_asc(self):
        df = pd.DataFrame({
            "text": ["A", "B", "C", "D"],
            "x": [50, 10, 30, 20],
            "y": [100, 100, 200, 200],
        })

        sorted_df = bbox_sort(df)

        self.assertEqual(list(sorted_df['text']), ['D', 'C', 'B', 'A'])

    def test_bbox_sort_resets_index(self):
        df = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4]
        }, index = [10, 20])

        sorted_df = bbox_sort(df)

        self.assertEqual(list(sorted_df.index), [0, 1])

    def test_sort_texts_single_doc_single_page(self):
        df = pd.DataFrame({
            "doc_id": [1, 1, 1],
            "page": [1, 1, 1],
            "text": ["A", "B", "C"],
            "bbox": [
                (0, 0, 10, 10),  # centroid (5, 5)
                (0, 20, 10, 30),  # centroid (5, 25)
                (20, 20, 30, 30),  # centroid (25, 25)
            ],
        })

        result = sort_texts(df)

        self.assertEqual(list(result["text"]), ["B", "C", "A"])

    def test_sort_texts_groups_by_doc_and_page(self):
        df = pd.DataFrame({
            "doc_id": [1, 1, 1, 1],
            "page":   [1, 1, 2, 2],
            "text":   ["A", "B", "C", "D"],
            "bbox": [
                (0, 0, 10, 10),    # page 1
                (0, 20, 10, 30),  # page 1
                (0, 0, 10, 10),   # page 2
                (0, 20, 10, 30),  # page 2
            ],
        })

        result = sort_texts(df)

        page1 = result[result["page"] == 1]["text"].tolist()
        page2 = result[result["page"] == 2]["text"].tolist()

        self.assertEqual(page1, ["B", "A"])
        self.assertEqual(page2, ["D", "C"])

    def test_sort_texts_preserves_row_count(self):
        df = pd.DataFrame({
            "doc_id": [1, 1],
            "page": [1, 1],
            "bbox": [(0, 0, 10, 10), (10, 10, 20, 20)],
        })

        result = sort_texts(df)

        self.assertEqual(len(result), len(df))

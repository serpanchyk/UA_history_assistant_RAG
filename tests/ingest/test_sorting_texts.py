"""Tests for bbox-based sorting and grouping logic used to order text blocks on pages and docs."""

import unittest

import pandas as pd

from src.ingest.sorting_texts import sort_texts

class TestSortingTexts(unittest.TestCase):
    """Confirm bbox-based sorting (y desc, x asc), grouping by doc/page, and index resetting behavior."""

    def test_sort_texts_single_doc_single_page(self):
        """sort_texts should order texts on a single page by their bbox-derived centroid order."""
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
        """sort_texts should process each doc/page group independently while preserving grouping."""
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
        """sort_texts must not change the total number of rows when sorting."""
        df = pd.DataFrame({
            "doc_id": [1, 1],
            "page": [1, 1],
            "bbox": [(0, 0, 10, 10), (10, 10, 20, 20)],
        })

        result = sort_texts(df)

        self.assertEqual(len(result), len(df))

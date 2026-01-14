import unittest
from unittest.mock import patch

import pandas as pd

from src.ingest.chunking import chunking


class TestChunking(unittest.TestCase):
    def test_text_gets_chunked_in_one_doc(self):
        text_blocks_df = pd.DataFrame({
            'text': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'doc_id': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            'page': [0, 0, 1, 1, 1, 2, 2, 2, 3],
        })

        texts = chunking(text_blocks_df, 6)

        self.assertEqual(len(texts), 3)
        self.assertEqual(texts[0]['text'], '1 2 3')
        self.assertEqual(texts[1]['text'], '4 5 6')
        self.assertEqual(texts[2]['text'], '7 8 9')
        self.assertEqual(texts[0]['pages'], [0, 1])
        self.assertEqual(texts[1]['pages'], [1, 2])
        self.assertEqual(texts[2]['pages'], [2, 3])
        self.assertEqual(texts[0]['doc_id'], 0)
        self.assertEqual(texts[1]['doc_id'], 0)
        self.assertEqual(texts[2]['doc_id'], 0)

    def test_text_gets_chunked_in_multiple_docs(self):

        text_blocks_df = pd.DataFrame({
            'text': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'doc_id': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'page': [0, 0, 1, 1, 1, 2, 2, 2, 3],
        })

        texts = chunking(text_blocks_df)

        self.assertEqual(len(texts), 3)
        self.assertEqual(texts[0]['text'], '1 2 3')
        self.assertEqual(texts[1]['text'], '4 5 6')
        self.assertEqual(texts[2]['text'], '7 8 9')
        self.assertEqual(texts[0]['pages'], [0, 1])
        self.assertEqual(texts[1]['pages'], [1, 2])
        self.assertEqual(texts[2]['pages'], [2, 3])
        self.assertEqual(texts[0]['doc_id'], 0)
        self.assertEqual(texts[1]['doc_id'], 1)
        self.assertEqual(texts[2]['doc_id'], 2)


if __name__ == '__main__':
    unittest.main()
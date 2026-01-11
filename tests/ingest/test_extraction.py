import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

from src.ingest.extraction import (
    extract_text_block,
    save_image,
    extract_image,
    iter_pdf_pages,
    iter_page_blocks,
    process_block,
    extract_blocks_from_pdf,
    get_pdf_path,
    extract_data, TEXT_BLOCK_TYPE, IMAGE_BLOCK_TYPE
)
from tests.fixtures.mock_classes import MockPage


class TestExtractionData(unittest.TestCase):
    def test_extract_text_block_returns_block(self):
        block = {'bbox': [0, 0, 100, 100]}
        with patch('src.ingest.extraction.block_to_text', return_value='Hello'):
            result = extract_text_block(block, doc_id=1, page_number=2)

        self.assertEqual(result['text'], 'Hello')
        self.assertEqual(result['bbox'], [0, 0, 100, 100])
        self.assertEqual(result['doc_id'], 1)
        self.assertEqual(result['page'], 2)

    def test_extract_text_block_none_when_no_text(self):
        block = {'bbox': [0, 0, 100, 100]}
        with patch('src.ingest.extraction.block_to_text', return_value=None):
            result = extract_text_block(block, doc_id=1, page_number=2)

        self.assertIsNone(result)

    @patch('src.ingest.extraction.write_image')
    def test_save_image_saves_and_returns_path(self, mock_write_image):
        image_bytes = b"fakebytes"
        ext = "png"
        doc_id = 1
        page_number = 2

        path = save_image(image_bytes, ext, doc_id, page_number)
        self.assertTrue(str(path).endswith(f"doc{doc_id}_page{page_number}_{path.stem[-8:]}.{ext}"))
        mock_write_image.assert_called_once_with(image_bytes, path)

    @patch('src.ingest.extraction.save_image', return_value=Path("/fake/path/image.png"))
    def test_extract_image_returns_info(self, mock_save_image):
        block = {"bbox": [0, 0, 10, 10], "image": b"imgbytes", "ext": "png"}
        result = extract_image(block, doc_id=1, page_number=2)
        self.assertIsNotNone(result)
        self.assertEqual(result['bbox'], [0, 0, 10, 10])
        self.assertEqual(result['page'], 2)
        self.assertEqual(result['doc_id'], 1)
        self.assertEqual(result['path'], "/fake/path/image.png")

    def test_process_block_text(self):
        block = {"type": TEXT_BLOCK_TYPE, "bbox": [0, 0, 10, 10]}
        with patch('src.ingest.extraction.extract_text_block', return_value={"text": "x"}):
            text, image = process_block(block, doc_id=1, page_number=1)
        self.assertEqual(text, {"text": "x"})
        self.assertIsNone(image)

    @patch('src.ingest.extraction.extract_image',
           return_value={"path": "/fake", "bbox": [0, 0, 1, 1], "page": 1, "doc_id": 1})
    def test_process_block_image(self, mock_extract_image):
        block = {"type": IMAGE_BLOCK_TYPE, "bbox": [0, 0, 1, 1], "image": b"bytes", "ext": "png"}
        text, image = process_block(block, doc_id=1, page_number=1)
        self.assertIsNone(text)
        self.assertEqual(image['path'], "/fake")

    @patch('src.ingest.extraction.TEXTBOOKS_DIR_PATH', Path("/fake/dir"))
    def test_get_pdf_path_exists(self):
        fake_pdf_name = "file.pdf"
        row = MagicMock()
        row.pdf_name = fake_pdf_name
        with patch('pathlib.Path.exists', return_value=True):
            path = get_pdf_path(row)
        self.assertTrue(str(path).endswith(fake_pdf_name))

    @patch('src.ingest.extraction.TEXTBOOKS_DIR_PATH', Path("/fake/dir"))
    def test_get_pdf_path_missing(self):
        row = MagicMock()
        row.pdf_name = "missing.pdf"
        with patch('pathlib.Path.exists', return_value=False):
            path = get_pdf_path(row)
        self.assertIsNone(path)

class TestEstractionDataIntegration(unittest.TestCase):

    @patch("src.ingest.extraction.pymupdf.open")
    def test_iter_pdf_pages_yields_pages(self, mock_open):
        mock_doc = MagicMock()
        mock_doc.__iter__.return_value = ["page1", "page2"]
        mock_open.return_value.__enter__.return_value = mock_doc

        pages = list(iter_pdf_pages(Path("/fake.pdf")))
        self.assertEqual(pages, ["page1", "page2"])
        mock_open.assert_called_once_with(Path("/fake.pdf"))

    def test_iter_page_blocks_returns_blocks(self):
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [{"type": 0}, {"type": 1}]
        }
        blocks = iter_page_blocks(page)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["type"], 0)
        self.assertEqual(blocks[1]["type"], 1)

    @patch("src.ingest.extraction.iter_pdf_pages")
    @patch("src.ingest.extraction.iter_page_blocks")
    @patch("src.ingest.extraction.process_block")
    def test_extract_blocks_from_pdf(self, mock_process_block, mock_iter_page_blocks, mock_iter_pdf_pages):
        mock_iter_pdf_pages.return_value = [MockPage(1)]
        mock_iter_page_blocks.return_value = [{"type": 0}, {"type": 1}]
        mock_process_block.side_effect = [
            ({"text": "t"}, None),
            (None, {"path": "/img.png"})
        ]

        texts, images = extract_blocks_from_pdf(Path("/fake.pdf"), doc_id=42)
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["text"], "t")
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["path"], "/img.png")

    @patch("src.ingest.extraction.extract_blocks_from_pdf")
    @patch("src.ingest.extraction.get_pdf_path")
    def test_extract_data_accumulates_all(self, mock_get_pdf_path, mock_extract_blocks_from_pdf):
        df_docs = pd.DataFrame({"pdf_name": ["a.pdf", "b.pdf"]})
        mock_get_pdf_path.side_effect = [Path("/fake/a.pdf"), Path("/fake/b.pdf")]
        mock_extract_blocks_from_pdf.side_effect = [
            ([{"text": "t1"}], [{"path": "/img1.png"}]),
            ([{"text": "t2"}], [{"path": "/img2.png"}])
        ]

        texts, images = extract_data(df_docs)
        self.assertEqual(len(texts), 2)
        self.assertEqual(texts[0]["text"], "t1")
        self.assertEqual(texts[1]["text"], "t2")
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["path"], "/img1.png")
        self.assertEqual(images[1]["path"], "/img2.png")


if __name__ == '__main__':
    unittest.main()
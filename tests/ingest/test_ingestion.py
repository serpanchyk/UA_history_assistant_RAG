import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from unittest.mock import ANY

from src.ingest.ingesting import PDFIngestor
from src import TEXTBOOKS_DF_PATH, TEXT_BLOCKS_DF_PATH, IMAGES_DF_PATH

class TestPDFIngestor(unittest.TestCase):
    def setUp(self):
        self.ingestor = PDFIngestor(images_dir=Path("/fake/images"))

    @patch("src.ingest.ingesting.delete_images")
    def test_delete_old_images_calls_delete(self, mock_delete):
        self.ingestor.delete_old_images(force=True)
        mock_delete.assert_called_once_with(Path("/fake/images"), force=True)

    @patch("src.ingest.ingesting.read_parquet")
    def test_load_textbooks_returns_dataframe(self, mock_read):
        mock_read.return_value = pd.DataFrame({"pdf_name": ["a.pdf", "b.pdf"]})
        df = self.ingestor.load_textbooks()
        self.assertEqual(len(df), 2)
        mock_read.assert_called_once_with(TEXTBOOKS_DF_PATH)

    @patch("src.ingest.ingesting.extract_data")
    def test_extract_data_from_pdfs_returns_dataframes(self, mock_extract):
        mock_extract.return_value = (
            [{"text": "t1"}, {"text": "t2"}],
            [{"path": "/img1.png"}, {"path": "/img2.png"}]
        )
        textbooks_df = pd.DataFrame({"pdf_name": ["a.pdf"]})
        texts_df, images_df = self.ingestor.extract_data_from_pdfs(textbooks_df)

        self.assertEqual(len(texts_df), 2)
        self.assertEqual(texts_df.iloc[0]["text"], "t1")
        self.assertEqual(len(images_df), 2)
        self.assertEqual(images_df.iloc[1]["path"], "/img2.png")

    @patch("src.ingest.ingesting.filter_images")
    def test_filter_images_df(self, mock_filter):
        images_df = pd.DataFrame({"path": ["/img1.png", "/img2.png"]})
        mock_filter.return_value = images_df.iloc[[0]]
        filtered = self.ingestor.filter_images_df(images_df)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["path"], "/img1.png")

    @patch("src.ingest.ingesting.link_image_to_text")
    def test_link_images_to_text(self, mock_link):
        images_df = pd.DataFrame({"path": ["/img1.png"]})
        texts_df = pd.DataFrame({"text": ["Hello"]})
        mock_link.return_value = images_df.assign(caption="Hello")
        linked = self.ingestor.link_images_to_text(images_df, texts_df)
        self.assertIn("caption", linked.columns)
        self.assertEqual(linked.iloc[0]["caption"], "Hello")

    @patch("src.ingest.ingesting.write_parquet")
    def test_save_results_calls_write_parquet(self, mock_write):
        texts_df = pd.DataFrame({"text": ["t"]})
        images_df = pd.DataFrame({"path": ["/img.png"]})
        self.ingestor.save_results(texts_df, images_df)
        mock_write.assert_any_call(ANY, TEXT_BLOCKS_DF_PATH)
        mock_write.assert_any_call(ANY, IMAGES_DF_PATH)

    @patch("src.ingest.ingesting.PDFIngestor.delete_old_images")
    @patch("src.ingest.ingesting.PDFIngestor.load_textbooks")
    @patch("src.ingest.ingesting.PDFIngestor.extract_data_from_pdfs")
    @patch("src.ingest.ingesting.PDFIngestor.filter_images_df")
    @patch("src.ingest.ingesting.PDFIngestor.link_images_to_text")
    @patch("src.ingest.ingesting.PDFIngestor.save_results")
    def test_run_pipeline_all_steps(self, mock_save, mock_link, mock_filter, mock_extract, mock_load, mock_delete):
        mock_load.return_value = pd.DataFrame({"pdf_name": ["a.pdf"]})
        mock_extract.return_value = (
            pd.DataFrame({"text": ["t"]}).to_dict("records"),
            pd.DataFrame({"path": ["/img.png"]}).to_dict("records")
        )
        mock_filter.return_value = pd.DataFrame({"path": ["/img.png"]})
        mock_link.return_value = pd.DataFrame({"path": ["/img.png"], "caption": ["t"]})

        self.ingestor.run(filter_images_flag=True, link_images_flag=True)

        mock_delete.assert_called_once()
        mock_load.assert_called_once()
        mock_extract.assert_called_once()
        mock_filter.assert_called_once()
        mock_link.assert_called_once()
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Tests for DataFrame <-> Parquet helpers: reading, writing and error/log behavior."""

import unittest
from unittest.mock import patch
from pathlib import Path
import pandas as pd

from src.fs_io.dataframes import read_parquet, write_parquet

class TestParquetUtils(unittest.TestCase):
    """Validate read/write helpers for DataFrame <-> Parquet including error logging and directory handling."""

    @patch("src.fs_io.dataframes.logger")
    def test_read_parquet_file_not_found(self, mock_logger):
        """Ensure FileNotFoundError is raised and an error is logged when parquet is missing."""
        fake_path = Path("/fake/file.parquet")
        with self.assertRaises(FileNotFoundError) as cm:
            read_parquet(fake_path)
        self.assertIn(str(fake_path), str(cm.exception))
        mock_logger.error.assert_called_once_with(f"Parquet file not found: {fake_path}")

    @patch("src.fs_io.dataframes.pd.read_parquet")
    @patch("src.fs_io.dataframes.Path.exists", return_value=True)
    @patch("src.fs_io.dataframes.logger")
    def test_read_parquet_success(self, mock_logger, mock_exists, mock_read_parquet):
        """Confirm successful read_parquet calls pd.read_parquet and logs info."""
        fake_path = Path("/fake/file.parquet")
        df_mock = pd.DataFrame({"a": [1, 2]})
        mock_read_parquet.return_value = df_mock

        result = read_parquet(fake_path)

        mock_read_parquet.assert_called_once_with(fake_path)
        pd.testing.assert_frame_equal(result, df_mock)
        mock_logger.info.assert_called_once_with(f"Reading Parquet file: {fake_path}")

    @patch("src.fs_io.dataframes.ensure_parent_dir")
    @patch("src.fs_io.dataframes.logger")
    def test_write_parquet_calls_methods(self, mock_logger, mock_ensure_parent_dir):
        """Verify write_parquet ensures parent dir and delegates to DataFrame.to_parquet."""
        fake_path = Path("/fake/dir/file.parquet")
        df_mock = pd.DataFrame({"a": [1, 2]})
        with patch.object(df_mock, "to_parquet") as mock_to_parquet:
            write_parquet(df_mock, fake_path)
            mock_ensure_parent_dir.assert_called_once_with(fake_path)
            mock_to_parquet.assert_called_once_with(fake_path)
            mock_logger.info.assert_called_once_with(f"Wrote DataFrame to Parquet file: {fake_path}")


if __name__ == "__main__":
    unittest.main()

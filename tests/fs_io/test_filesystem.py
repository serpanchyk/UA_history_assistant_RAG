"""Tests for filesystem utilities that create directories and prepare parent folders for files."""

import unittest
from unittest.mock import patch
from pathlib import Path

from src.fs_io.filesystem import ensure_dir, ensure_parent_dir


class TestFsUtils(unittest.TestCase):
    """Ensure directory utilities create and reference parent directories correctly and log actions."""

    @patch("src.fs_io.filesystem.logger")
    def test_ensure_dir_creates_dir(self, mock_logger):
        """ensure_dir should call Path.mkdir with parents=True and exist_ok=True and log debug."""
        path = Path("/fake/dir/path")
        with patch.object(Path, "mkdir") as mock_mkdir:
            ensure_dir(path)
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_logger.debug.assert_called_once_with(f"Ensured directory exists: {path}")

    @patch("src.fs_io.filesystem.ensure_dir")
    @patch("src.fs_io.filesystem.logger")
    def test_ensure_parent_dir_calls_ensure_dir(self, mock_logger, mock_ensure_dir):
        """ensure_parent_dir should delegate to ensure_dir using the file's parent directory."""
        file_path = Path("/fake/dir/file.txt")
        ensure_parent_dir(file_path)
        mock_ensure_dir.assert_called_once_with(file_path.parent)


if __name__ == "__main__":
    unittest.main()

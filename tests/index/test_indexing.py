"""Tests for run_indexing: verify how chunks and images are discovered and indexed into the vector store."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from src.index.indexing import run_indexing

class TestRunIndexing(unittest.TestCase):
    """Test suite ensuring run_indexing correctly handles presence/absence of chunks and images."""

    def setUp(self):
        """Set up patched paths, readers, and a mock vector store for each test."""
        self.patches = {
            "chunks_path": patch("src.index.indexing.CHUNKS_DF_PATH"),
            "images_path": patch("src.index.indexing.IMAGES_DF_PATH"),
            "read_parquet": patch("src.index.indexing.read_parquet"),
            "embedder_cls": patch("src.index.indexing.EmbeddingService"),
            "vector_store_cls": patch("src.index.indexing.QdrantVectorStore"),
            "logger": patch("src.index.indexing.logger"),
        }

        self.mocks = {}
        for name, patcher in self.patches.items():
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            self.mocks[name] = mock

        self.vector_store = MagicMock()
        self.mocks["vector_store_cls"].return_value = self.vector_store

    def test_index_both_exist(self):
        """When both chunks and images exist, both should be added to the vector store."""
        self.mocks["chunks_path"].exists.return_value = True
        self.mocks["images_path"].exists.return_value = True

        chunks_df = pd.DataFrame([{"text": "hello"}])
        images_df = pd.DataFrame([{"image": "img"}])

        self.mocks["read_parquet"].side_effect = [chunks_df, images_df]

        run_indexing()

        self.vector_store.add_text_entry.assert_called_once_with(
            chunks_df.to_dict(orient="records")
        )
        self.vector_store.add_image_entry.assert_called_once_with(images_df)

    def test_only_chunks_exist(self):
        """When only chunks exist, only text entries should be added to the vector store."""
        self.mocks["chunks_path"].exists.return_value = True
        self.mocks["images_path"].exists.return_value = False

        chunks_df = pd.DataFrame([{"text": "only chunks"}])
        self.mocks["read_parquet"].return_value = chunks_df

        run_indexing()

        self.vector_store.add_text_entry.assert_called_once()
        self.vector_store.add_image_entry.assert_not_called()

    def test_only_images_exist(self):
        """When only images exist, only image entries should be added to the vector store."""
        self.mocks["chunks_path"].exists.return_value = False
        self.mocks["images_path"].exists.return_value = True

        images_df = pd.DataFrame([{"image": "only image"}])
        self.mocks["read_parquet"].return_value = images_df

        run_indexing()

        self.vector_store.add_text_entry.assert_not_called()
        self.vector_store.add_image_entry.assert_called_once_with(images_df)

    def test_no_files_exist(self):
        """When no source files exist, nothing should be added and no reads performed."""
        self.mocks["chunks_path"].exists.return_value = False
        self.mocks["images_path"].exists.return_value = False

        run_indexing()

        self.vector_store.add_text_entry.assert_not_called()
        self.vector_store.add_image_entry.assert_not_called()
        self.mocks["read_parquet"].assert_not_called()

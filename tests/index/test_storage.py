import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from src.index.storage import QdrantVectorStore


class TestQdrantStorageReal(unittest.TestCase):
    """
    Test suite for QdrantVectorStore focusing on hybrid retrieval,
    data integrity, and indexing workflow.
    """

    def setUp(self):
        self.real_client = QdrantClient(":memory:")
        self.mock_embedder = MagicMock()
        self.mock_embedder.DENSE_DIM = 4
        self.mock_embedder.CLIP_DIM = 4

        default_vector = {
            "dense": [0.1] * 4,
            "sparse": {"indices": [0], "values": [0.9]},
            "image": [0.5] * 4
        }
        self.mock_embedder.embed_hybrid.return_value = default_vector
        self.mock_embedder.embed_text.return_value = {
            k: v for k, v in default_vector.items() if k != "image"
        }

        self.store = QdrantVectorStore(
            embedding_service=self.mock_embedder,
            client=self.real_client
        )

        self.patch_path = "src.index.storage"
        patches = [
            patch(f"{self.patch_path}.get_textbook_source", side_effect=lambda x: f"source_{x}"),
            patch(f"{self.patch_path}.list_to_interval",
                  side_effect=lambda x: f"{x[0]}-{x[-1]}" if isinstance(x, list) else str(x)),
            patch(f"{self.patch_path}.CHUNKS_DF_PATH"),
            patch(f"{self.patch_path}.IMAGES_DF_PATH"),
            patch(f"{self.patch_path}.read_parquet"),
            patch(f"{self.patch_path}.logger"),
        ]

        self.mocks = {}
        for p in patches:
            name = p.attribute if hasattr(p, 'attribute') else p.target.split('.')[-1]
            self.mocks[name] = p.start()
            self.addCleanup(p.stop)

    # --- Formatting Tests ---

    def test_text_context_formatting(self):
        """Verifies text document conversion to LLM context string."""
        doc = Document(page_content="Txt", metadata={'doc_id': 1, 'pages': [1, 3]})
        result = QdrantVectorStore.text_documents_to_llm_context([doc])
        self.assertIn("Джерело: source_1", result)
        self.assertIn("Сторінки: 1-3", result)
        self.assertIn("Контекст: Txt", result)

    def test_image_context_formatting(self):
        """Verifies image document conversion to LLM context string."""
        doc = Document(page_content="Img", metadata={'doc_id': 2, 'page': 5})
        result = QdrantVectorStore.image_documents_to_llm_context([doc])
        self.assertIn("Джерело: source_2", result)
        self.assertIn("Сторінка: 5", result)
        self.assertIn("Опис зображення: Img", result)

    def test_empty_context_formatting(self):
        """Ensures empty document lists return empty strings."""
        self.assertEqual(QdrantVectorStore.text_documents_to_llm_context([]), "")
        self.assertEqual(QdrantVectorStore.image_documents_to_llm_context([]), "")

    # --- Database & Retrieval Tests ---

    def test_collection_schema_initialization(self):
        """Checks if Qdrant collections are created with correct vector dimensions."""
        img_cfg = self.real_client.get_collection(self.store.IMAGE_COLLECTION).config.params.vectors
        txt_cfg = self.real_client.get_collection(self.store.TEXT_COLLECTION).config.params.vectors
        self.assertEqual(img_cfg['dense'].size, 4)
        self.assertEqual(img_cfg['image'].size, 4)
        self.assertEqual(txt_cfg['dense'].size, 4)

    def test_image_insertion_integrity(self):
        """Verifies that image data and vectors are correctly persisted."""
        data = [{"caption": "test", "path": "p.jpg", "doc_id": "d1", "page": 1}]
        self.store.add_image_entry(data)
        points = self.real_client.scroll(self.store.IMAGE_COLLECTION, with_vectors=True)[0]
        self.assertEqual(len(points), 1)
        self.assertIn("image", points[0].vector)
        self.assertEqual(points[0].payload['caption'], "test")

    def test_text_insertion_integrity(self):
        """Verifies that text chunks are correctly persisted."""
        data = [{"text": "hello", "pages": [1], "doc_id": "t1"}]
        self.store.add_text_entry(data)
        points = self.real_client.scroll(self.store.TEXT_COLLECTION)[0]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].payload['text'], "hello")

    def test_hybrid_retrieve_all(self):
        """Tests the unified retrieval pipeline for texts and images."""
        self.store.add_text_entry([{"text": "val", "pages": [1], "doc_id": "t1"}])
        results = self.store.retrieve_all("query", k_text=1)
        self.assertEqual(len(results['texts']), 1)
        self.assertEqual(results['texts'][0].page_content, "val")

    # --- Run Workflow Tests ---

    def _mock_run_dependencies(self, chunks_exist: bool, images_exist: bool):
        """Configures file path existence and wraps entry methods in mocks."""
        self.mocks["CHUNKS_DF_PATH"].exists.return_value = chunks_exist
        self.mocks["IMAGES_DF_PATH"].exists.return_value = images_exist
        self.store.add_text_entry = MagicMock()
        self.store.add_image_entry = MagicMock()

    def test_run_with_full_data(self):
        """Verifies indexing when both text and image source files exist."""
        self._mock_run_dependencies(chunks_exist=True, images_exist=True)
        self.mocks["read_parquet"].side_effect = [pd.DataFrame([{"text": "a"}]), pd.DataFrame([{"image": "b"}])]
        self.store.run()
        self.store.add_text_entry.assert_called_once()
        self.store.add_image_entry.assert_called_once()

    def test_run_with_partial_data(self):
        """Verifies indexing when only text source files exist."""
        self._mock_run_dependencies(chunks_exist=True, images_exist=False)
        self.mocks["read_parquet"].return_value = pd.DataFrame([{"text": "a"}])
        self.store.run()
        self.store.add_text_entry.assert_called_once()
        self.store.add_image_entry.assert_not_called()

    def test_run_with_no_data(self):
        """Verifies run does nothing when no source files are found."""
        self._mock_run_dependencies(chunks_exist=False, images_exist=False)
        self.store.run()
        self.mocks["read_parquet"].assert_not_called()
        self.store.add_text_entry.assert_not_called()

    def test_empty_input_resilience(self):
        """Ensures entry methods do not raise exceptions on empty input lists."""
        self.store.add_text_entry([])
        self.store.add_image_entry([])
        txt_count = self.real_client.count(self.store.TEXT_COLLECTION).count
        self.assertEqual(txt_count, 0)

if __name__ == "__main__":
    unittest.main()
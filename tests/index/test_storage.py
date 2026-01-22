import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from src.utils.texts import get_textbook_source
from src.index.storage import QdrantVectorStore


class TestQdrantStorageReal(unittest.TestCase):

    def setUp(self):
        self.real_client = QdrantClient(":memory:")
        self.mock_embedder = MagicMock()
        self.mock_embedder.DENSE_DIM = 4
        self.mock_embedder.CLIP_DIM = 4

        self.mock_embedder.embed_hybrid.return_value = {
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": {"indices": [0], "values": [0.9]},
            "image": [0.5, 0.6, 0.7, 0.8]
        }

        self.mock_embedder.embed_text.return_value = {
            "dense": [0.1, 0.2, 0.3, 0.4],
            "sparse": {"indices": [0], "values": [0.9]}
        }

        self.store = QdrantVectorStore(
            embedding_service=self.mock_embedder,
            client=self.real_client
        )

        patcher = patch("src.index.storage.get_textbook_source")
        self.mock_get_source = patcher.start()
        self.mock_get_source.side_effect = lambda doc_id: f"source_{doc_id}"
        self.addCleanup(patcher.stop)

    def make_doc(self, metadata: dict, content="dummy content"):
        return Document(
            page_content=content,
            metadata=metadata
        )

    def test_text_documents_to_llm_context_basic(self):
        docs = [
            self.make_doc({'doc_id': 1, 'pages': [1, 2, 3]}, "Content A"),
            self.make_doc({'doc_id': 2, 'pages': [4, 5]}, "Content B")
        ]

        result = QdrantVectorStore.text_documents_to_llm_context(docs)
        self.assertIn("[ДОКУМЕНТ]", result)
        self.assertIn("Джерело: source_1", result)
        self.assertIn("Джерело: source_2", result)
        self.assertIn("Контекст: Content A", result)
        self.assertIn("Контекст: Content B", result)
        self.assertIn('Сторінки: 1-3', result)
        self.assertIn('Сторінки: 4-5', result)
        self.assertIn("\n---\n", result)

    def test_image_documents_to_llm_context_basic(self):
        docs = [
            self.make_doc({'doc_id': 1, 'page': 1}, "bad content"),
            self.make_doc({'doc_id': 2, 'page': 2}, 'good content')
        ]

        result = QdrantVectorStore.image_documents_to_llm_context(docs)
        self.assertIn("[ОПИС ЗОБРАЖЕННЯ]", result)
        self.assertIn("Джерело: source_1", result)
        self.assertIn("Джерело: source_2", result)
        self.assertIn('Сторінка: 1', result)
        self.assertIn('Сторінка: 2', result)
        self.assertIn('Опис зображення: bad content', result)
        self.assertIn('Опис зображення: good content', result)
        self.assertIn("\n---\n", result)

    def test_empty_document_list_returns_empty_string(self):
        self.assertEqual(QdrantVectorStore.text_documents_to_llm_context([]), "")
        self.assertEqual(QdrantVectorStore.image_documents_to_llm_context([]), "")

    def test_collections_created_with_correct_config(self):
        """Verifies that collections physically exist in memory with the right schema."""
        img_info = self.real_client.get_collection(self.store.IMAGE_COLLECTION)
        vectors = img_info.config.params.vectors

        self.assertEqual(vectors['dense'].size, 4)
        self.assertEqual(vectors['image'].size, 4)

        txt_info = self.real_client.get_collection(self.store.TEXT_COLLECTION)
        self.assertEqual(txt_info.config.params.vectors['dense'].size, 4)

    def test_add_image_entry_data_integrity(self):
        """
        Verifies that data uploaded via add_image_entry lands correctly
        in the database with the CORRECT vector names.
        """
        df = pd.DataFrame([{
            "caption": "test_img", "path": "p.jpg",
            "doc_id": "d1", "page": 1
        }])

        self.store.add_image_entry(df)

        points, _ = self.real_client.scroll(
            collection_name=self.store.IMAGE_COLLECTION,
            limit=1,
            with_vectors=True
        )

        self.assertEqual(len(points), 1)
        point = points[0]

        self.assertEqual(point.payload['caption'], "test_img")
        self.assertEqual(point.payload['doc_id'], "d1")

        self.assertIn("dense", point.vector)
        self.assertIn("image", point.vector)
        self.assertIn("sparse", point.vector)

    def test_add_text_entry_simple(self):
        chunks = [{"text": "hello world", "pages": [1], "doc_id": "d2"}]
        self.store.add_text_entry(chunks)

        points, _ = self.real_client.scroll(
            collection_name=self.store.TEXT_COLLECTION,
            limit=1
        )

        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].payload['text'], "hello world")

    def test_retrieve_all_returns_results(self):
        self.store.add_text_entry([{"text": "found me", "pages": [1], "doc_id": "find_1"}])

        results = self.store.retrieve_all("query", k_text=1)

        texts = results['texts']
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0].page_content, "found me")
        self.assertEqual(texts[0].metadata['doc_id'], "find_1")

    def test_edge_case_empty_inputs(self):
        empty_df = pd.DataFrame(columns=["caption", "path", "doc_id", "page"])

        try:
            self.store.add_image_entry(empty_df)
            self.store.add_text_entry([])
        except Exception as e:
            self.fail(f"Adding empty data raised exception: {e}")

        count = self.real_client.count(self.store.IMAGE_COLLECTION).count
        self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
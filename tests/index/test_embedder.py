import unittest
import numpy as np
from unittest.mock import  patch
from pathlib import Path


from src.index.embedder import EmbeddingService

class TestEmbeddingService(unittest.TestCase):

    def setUp(self):
        """Runs before each test. Sets up common mocks."""
        # Create a patcher for the dependencies
        self.bge_patcher = patch('src.index.embedder.BGEM3FlagModel')
        self.clip_patcher = patch('src.index.embedder.SentenceTransformer')
        self.read_img_patcher = patch('src.index.embedder.read_image')
        self.cv2pil_patcher = patch('src.index.embedder.cv2_array_to_PIL')

        # Start patchers
        self.mock_bge_cls = self.bge_patcher.start()
        self.mock_clip_cls = self.clip_patcher.start()
        self.mock_read_img = self.read_img_patcher.start()
        self.mock_cv2pil = self.cv2pil_patcher.start()

        # Configure BGE Mock
        self.mock_bge_inst = self.mock_bge_cls.return_value
        self.mock_bge_inst.model.model.config.hidden_size = 1024
        self.mock_bge_inst.tokenizer.vocab_size = 250002

        # Configure CLIP Mock
        self.mock_clip_inst = self.mock_clip_cls.return_value

        # Instantiate service
        self.service = EmbeddingService(device="cpu")

    def tearDown(self):
        """Stops all patchers."""
        patch.stopall()

    def test_initialization(self):
        """Test if models are loaded with correct args."""

        self.mock_bge_cls.assert_called_with(
            'BAAI/bge-m3',
            device='cpu',
            use_fp16=True,
            max_tokens=1024
        )
        self.assertEqual(self.service.DENSE_DIM, 1024)

    def test_embed_text_structure(self):
        """Test dense/sparse return types and sparse key conversion."""
        mock_output = {
            "dense_vecs": np.random.rand(1024),
            "lexical_weights": [{"101": 0.5, "202": 0.3}]
        }
        self.mock_bge_inst.encode.return_value = mock_output

        dense, sparse = self.service.embed_text("Test Query")

        self.assertIsInstance(dense, list)
        self.assertEqual(len(dense), 1024)

        self.assertIsInstance(sparse, dict)
        first_key = next(iter(sparse.keys()))
        self.assertIsInstance(first_key, int)
        self.assertEqual(sparse[101], 0.5)

    def test_embed_image(self):
        """Test image embedding pipeline."""
        self.mock_clip_inst.encode.return_value = np.random.rand(512)
        path = Path("test.jpg")

        vector = self.service.embed_image(path)

        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 512)
        self.mock_read_img.assert_called_once_with(path)
        self.mock_cv2pil.assert_called_once()

    def test_embed_hybrid(self):
        """Test hybrid wrapper calls both underlying methods."""
        self.mock_bge_inst.encode.return_value = {
            "dense_vecs": np.zeros(1024),
            "lexical_weights": [{"1": 0.1}]
        }
        self.mock_clip_inst.encode.return_value = np.zeros(512)

        result = self.service.embed_hybrid("text", Path("img.png"))

        keys = ["caption_dense_vector", "caption_sparse_vector", "image_vector"]
        for k in keys:
            self.assertIn(k, result)

        self.assertEqual(len(result["caption_dense_vector"]), 1024)

    def test_sparse_vector_edge_case_single_dict(self):
        """Test edge case where lexical_weights is a dict, not a list of dicts."""
        mock_output = {
            "dense_vecs": np.random.rand(1024),
            "lexical_weights": {"505": 0.9}
        }
        self.mock_bge_inst.encode.return_value = mock_output

        _, sparse = self.service.embed_text("Test")

        self.assertEqual(sparse[505], 0.9)


if __name__ == '__main__':
    unittest.main()
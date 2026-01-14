from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

from src import logger
from src.fs_io.images import read_image, cv2_array_to_PIL


class EmbeddingService:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding models on {self.device}...")

        # 1. Text Model (BGE-M3)
        self.text_model = BGEM3FlagModel(
            'BAAI/bge-m3',
            device=self.device,
            use_fp16=True,
        )

        # 2. Image Model (Multilingual CLIP)
        self.clip_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32-multilingual-v1',
            device=self.device
        )

        logger.info("Models loaded successfully.")

    def embed_text_dense_vector(self, text: str) -> List[float]:
        """Generates dense vector for text (1024 dim)."""
        # BGE-M3 returns a numpy array, convert to list for Qdrant
        output = self.text_model.encode(
            text,
            convert_to_tensor=False,
            return_dense=True,
            return_sparse=True,
        )
        return output["dense_vecs"].tolist()

    def embed_text_sparse_vector(self, text: str) -> List[float]:
        """Generates sparse vector for text (up to 1 million dim)."""
        output = self.text_model.encode(
            text,
            convert_to_tensor=False,
            return_dense=False,
            return_sparse=True,
        )

        embedding = list(
            map(lambda x: normalize(x, norm='l2'),
                map(dict_to_csr, output["lexical_weights"])
                )
        )
        return embedding


    def dict_to_csr(sparse_dict: dict) -> csr_matrix:
        length = len(sparse_dict)
        data = list(sparse_dict.values())
        cols = list(sparse_dict.keys())
        rows = np.zeros(length)

        return csr_matrix((data, (rows, cols)), shape=(1, VOCAB_SIZE), dtype=float)

    def embed_image(self, image_input: Path) -> List[float]:
        """Generates vector for image (512 dim). input is Path."""
        image_input = cv2_array_to_PIL(read_image(image_input))

        embedding = self.clip_model.encode(image_input, convert_to_tensor=False)

        return embedding.tolist()

    def embed_hybrid(self, text: str) -> Dict[str, List[float]]:
        """
        Creates TWO vectors for the same text (Caption).
        1. Semantic meaning (BGE-M3)
        2. Visual description meaning (CLIP Text Encoder)
        """
        return {
            "caption_dense_vector": self.embed_text_dense_vector(text),
            "caption_sparse_vector": self.embed_text_sparse_vector(text),
            "image_vector": self.clip_model.encode(text).tolist()
        }
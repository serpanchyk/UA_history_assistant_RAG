import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

from src import logger


class EmbeddingService:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading embedding models on {self.device}...")

        # 1. Text Model (BGE-M3)
        self.text_model = SentenceTransformer('BAAI/bge-m3', device=self.device)

        # 2. Image Model (Multilingual CLIP)
        self.clip_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device=self.device)

        logger.info("Models loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        """Generates dense vector for text (1024 dim)."""
        # BGE-M3 returns a numpy array, convert to list for Qdrant
        embedding = self.text_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_image(self, image_input: np.ndarray) -> List[float]:
        """Generates vector for image (512 dim). input is cv2 array."""


        embedding = self.clip_model.encode(image_input, convert_to_tensor=False)
        return embedding.tolist()

    def embed_hybrid(self, text: str) -> Dict[str, List[float]]:
        """
        Creates TWO vectors for the same text (Caption).
        1. Semantic meaning (BGE-M3)
        2. Visual description meaning (CLIP Text Encoder)
        """
        return {
            "caption_vector": self.embed_text(text),
            "image_vector": self.clip_model.encode(text).tolist()
        }
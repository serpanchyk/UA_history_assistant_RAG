from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import torch
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from typing import List, Dict

from src.logger import logger
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
            max_tokens= 1024,
        )

        # 2. Image Model (Multilingual CLIP)
        self.clip_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32-multilingual-v1',
            device=self.device
        )

        self.DENSE_DIM = self.text_model.model.model.config.hidden_size #1024
        self.VOCAB_SIZE = self.text_model.tokenizer.vocab_size # 250002
        self.CLIP_DIM = self.clip_model.get_sentence_embedding_dimension() #512

        logger.info("Models loaded successfully.")


    def _sparse_vector_to_qdrant(self, model_output: dict):
        """Extracts sparse weights and ensures format {int: float}."""
        raw_weights = model_output["lexical_weights"][0] \
            if isinstance(model_output["lexical_weights"], list) \
            else model_output["lexical_weights"]

        sorted_items = sorted(raw_weights.items(), key=lambda item: int(item[0]))

        indices, values = zip(*sorted_items) if sorted_items else ([], [])

        return {
            'indices': list(map(int, indices)),
            'values': list(map(float, values))
        }

    def embed_text(self, text: str):
        """Generates 2 vecs for text: dense and sparse embeddings"""
        bge_output = self.text_model.encode(
            text,
            convert_to_tensor=False,
            return_dense=True,
            return_sparse=True,
        )

        dense_vec = bge_output["dense_vecs"].tolist()

        sparse_vec = self._sparse_vector_to_qdrant(bge_output)

        return {
            "text_dense_vector": dense_vec,
            "text_sparse_vector": sparse_vec,
        }

    def embed_image(self, query: Path | str) -> List[float]:
        """Generates vector (512 dim) for image if Path were given or for text if str were given."""
        if isinstance(query, Path):
            image_input = cv2_array_to_PIL(read_image(query))
        elif isinstance(query, str):
            image_input = query
        else:
            raise ValueError("Query must be a Path or str representing an image or text.")

        embedding = self.clip_model.encode(image_input, convert_to_tensor=False)

        return embedding.tolist()

    def embed_hybrid(self, text: str, image: Path = None) -> Dict[str, List[float]]:
        """Creates vectors for a multimodal entry (Caption + Image)."""
        text_output = self.embed_text(text)
        dense_vec, sparse_vec =  text_output["text_dense_vector"], text_output["text_sparse_vector"]

        if image is None:
            image_vec = self.embed_image(text)
        else:
            image_vec = self.embed_image(image)

        return {
            "dense": dense_vec,
            "sparse": models.SparseVector(**sparse_vec),
            "image": image_vec
        }
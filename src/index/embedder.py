from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import torch
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from src.logger import logger
from src.fs_io.images import read_image, cv2_array_to_PIL

from enum import Enum

class EmbeddingMode(Enum):
    INDEX = "index"
    QUERY = "query"


class EmbeddingService:
    def __init__(self):
        gpu_available = torch.cuda.is_available()
        self.device = 'cuda' if gpu_available else 'cpu'
        logger.info(f"Loading embedding models on {self.device}...")

        self.max_tokens = 1024
        # 1. Text Model (BGE-M3)
        self.text_model = BGEM3FlagModel(
            'BAAI/bge-m3',
            device=self.device,
            use_fp16=gpu_available,
            max_tokens= self.max_tokens,
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


    def _format_sparse(self, model_output: dict) -> dict[str, list]:
        """Helper to format sparse output into {indices: [], values: []}."""
        raw_weights = model_output["lexical_weights"][0] \
            if isinstance(model_output["lexical_weights"], list) \
            else model_output["lexical_weights"]

        sorted_items = sorted((int(k), float(v)) for k, v in raw_weights.items())
        indices, values = zip(*sorted_items) if sorted_items else ([], [])

        return {
            'indices': indices,
            'values': values
        }

    def embed_text(self, text: str):
        """Generates 2 vecs for text: dense and sparse embeddings"""
        bge_output = self.text_model.encode(
            text,
            return_dense=True,
            return_sparse=True,
            show_progress_bar=True
        )

        dense_vec = bge_output["dense_vecs"].tolist()

        sparse_vec = self._format_sparse(bge_output)

        return {
            "dense": dense_vec,
            "sparse": sparse_vec,
        }

    def embed_image(self, query: Path | str) -> list[float]:
        """Generates vector (512 dim) for image if Path were given or for text if str were given."""
        if isinstance(query, Path):
            image_input = cv2_array_to_PIL(read_image(query))
        elif isinstance(query, str):
            image_input = query
        else:
            raise ValueError("Query must be a Path or str representing an image or text.")

        embedding = self.clip_model.encode(
            image_input,
            show_progress_bar=True
        )

        return embedding.tolist()

    def embed_hybrid(
        self,
        text: str,
        image: Path | None = None,
        mode: EmbeddingMode = EmbeddingMode.QUERY
    ) -> dict[str, list[float]]:
        """Creates vectors for a multimodal entry (Caption + Image)."""

        if mode is EmbeddingMode.INDEX:
            if image is None:
                raise ValueError("Image must be provided in INDEX mode.")
            image_vec = self.embed_image(image)
        elif mode is EmbeddingMode.QUERY:
            image_vec = self.embed_image(text)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        text_vecs = self.embed_text(text)


        return {
            "dense": text_vecs["dense"],
            "sparse": text_vecs["sparse"],
            "image": image_vec
        }
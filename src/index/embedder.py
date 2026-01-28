from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import torch
from sentence_transformers import SentenceTransformer

from enum import Enum
import os

from src.logger import logger
from src.fs_io.images import read_image, cv2_array_to_PIL
from src import MODELS_DIR_PATH

class EmbeddingMode(Enum):
    INDEX = "index"
    QUERY = "query"


class EmbeddingService:
    def __init__(self):

        self.cache_dir = MODELS_DIR_PATH
        os.makedirs(self.cache_dir, exist_ok=True)

        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(self.cache_dir)

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
            cache_dir = str(self.cache_dir)
        )

        # 2. Image Model (Multilingual CLIP)
        self.clip_model = SentenceTransformer(
            'sentence-transformers/clip-ViT-B-32-multilingual-v1',
            device=self.device,
            cache_folder = str(self.cache_dir)
        )

        self.DENSE_DIM = self.text_model.model.model.config.hidden_size #1024
        self.VOCAB_SIZE = self.text_model.tokenizer.vocab_size # 250002
        self.CLIP_DIM = self.clip_model.get_sentence_embedding_dimension() #512

        logger.info("Models loaded successfully.")


    def _format_sparse(self, raw_weights: dict) -> dict[str, list]:
        """Format a single dictionary of lexical weights."""
        if not raw_weights:
            return {'indices': [], 'values': []}

        sorted_items = sorted((int(k), float(v)) for k, v in raw_weights.items())
        indices, values = zip(*sorted_items)

        return {
            'indices': list(indices),
            'values': list(values)
        }

    def embed_text(self, text: str):
        """Generates 2 vecs for text: dense and sparse embeddings"""
        bge_output = self.text_model.encode(
            text,
            return_dense=True,
            return_sparse=True,
        )

        raw_sparse = bge_output["lexical_weights"]
        sparse_dict = raw_sparse[0] if isinstance(raw_sparse, list) else raw_sparse

        return {
            "dense": bge_output["dense_vecs"].tolist(),
            "sparse": self._format_sparse(sparse_dict),
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

    def embed_text_batch(self, texts: list[str]):
        """Generates dense and sparse embeddings for a batch of strings."""
        bge_output = self.text_model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
        )

        dense_vecs = bge_output["dense_vecs"].tolist()

        # Format sparse vectors for each item in the batch
        return {
            "dense": dense_vecs,
            "sparse": [self._format_sparse(vec) for vec in bge_output["lexical_weights"]],
        }

    def embed_image_batch(self, queries: list[Path | str]) -> list[list[float]]:
        """Generates vectors for a batch of images or text queries."""
        inputs = []
        for q in queries:
            if isinstance(q, Path):
                inputs.append(cv2_array_to_PIL(read_image(q)))
            else:
                inputs.append(q)

        embeddings = self.clip_model.encode(
            inputs,
            batch_size=len(inputs),
            show_progress_bar=False
        )
        return embeddings.tolist()
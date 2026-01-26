from pathlib import Path
from typing import Any, Iterable, Callable


from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models

from dotenv import load_dotenv
import os
import hashlib
import uuid
import json
from tqdm import tqdm
import numpy as np

from src import CHUNKS_DF_PATH, IMAGES_DF_PATH
from src.fs_io.dataframes import read_parquet
from src.fs_io.images import read_image
from src.index.embedder import EmbeddingMode
from src.logger import logger
from src.utils.texts import get_textbook_source, list_to_interval, sanitize

load_dotenv()

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embedding_service,
        client: QdrantClient = None,
    ):

        self.client = client if client is not None else  QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.embedding_service = embedding_service

        self.TEXT_COLLECTION = "ukrainian_historical_text"
        self.IMAGE_COLLECTION = "ukrainian_historical_image"

        self._init_collections()

    def _init_collections(self):
        collections_config = {
            self.IMAGE_COLLECTION: {
                "vectors_config": {
                    'dense': models.VectorParams(
                        size=self.embedding_service.DENSE_DIM,
                        distance=models.Distance.COSINE
                    ),
                    'image': models.VectorParams(
                        size=self.embedding_service.CLIP_DIM,
                        distance=models.Distance.COSINE
                    ),
                },
                "sparse_vectors_config": {
                    'sparse': models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    ),
                }
            },
            self.TEXT_COLLECTION: {
                "vectors_config": {
                    'dense': models.VectorParams(
                        size=self.embedding_service.DENSE_DIM,
                        distance=models.Distance.COSINE
                    ),
                },
                "sparse_vectors_config": {
                    'sparse': models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    ),
                }
            }
        }

        for name, config in collections_config.items():
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=config["vectors_config"],
                    sparse_vectors_config=config["sparse_vectors_config"]
                )

    def get_point(self, embeddings: dict, metadata: dict) -> models.PointStruct:
        vector = {}
        for name, embedding in embeddings.items():
            if 'sparse' in name:
                vector[name] = models.SparseVector(
                    indices=list(embedding["indices"]),
                    values=list(embedding["values"])
                )
            else:
                vector[name] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding

        clean_metadata = sanitize(metadata)

        content_str = json.dumps(
            clean_metadata,
            sort_keys=True,
            ensure_ascii=False,
        )

        hash_bytes = hashlib.sha256(content_str.encode('utf-8')).digest()
        point_id = str(uuid.UUID(bytes=hash_bytes[:16]))

        return models.PointStruct(
            id=point_id,
            vector=vector,
            payload=clean_metadata
        )

    def _process_and_upload(
            self,
            collection_name: str,
            items: list[Any],
            processor: Callable[[list[Any]], list[models.PointStruct]],
            batch_size: int = 64
    ):
        if not items:
            return

        for i in tqdm(range(0, len(items), batch_size), desc=f"Indexing {collection_name}"):
            batch_items = items[i: i + batch_size]

            points = processor(batch_items)

            self.client.upload_points(
                collection_name=collection_name,
                points=points,
                wait=True
            )

    def add_image_entry(self, images: list[dict]):
        def batch_image_processor(batch_list):
            paths = [img['path'] for img in batch_list]
            captions = [img['caption'] for img in batch_list]

            text_embeddings = self.embedding_service.embed_text_batch(captions)
            image_embeddings = self.embedding_service.embed_image_batch(paths)

            points = []
            for i, item in enumerate(batch_list):
                embeddings = {
                    "dense": text_embeddings["dense"][i],
                    "sparse": text_embeddings["sparse"][i],
                    "image": image_embeddings[i]
                }
                metadata = {
                    "caption": item['caption'],
                    "path": str(item['path']),
                    "doc_id": item['doc_id'],
                    "page": item['page']
                }
                points.append(self.get_point(embeddings, metadata))
            return points

        self._process_and_upload(
            collection_name=self.IMAGE_COLLECTION,
            items=images,
            processor=batch_image_processor,
        )

    def add_text_entry(self, text_chunks: list[dict]):
        def batch_text_processor(batch_list):
            texts = [row["text"] for row in batch_list]
            embeddings_batch = self.embedding_service.embed_text_batch(texts)

            points = []
            for i, item in enumerate(batch_list):
                embeddings = {
                    "dense": embeddings_batch["dense"][i],
                    "sparse": embeddings_batch["sparse"][i],
                }
                metadata = {
                    "text": item["text"],
                    "pages": item["pages"],
                    "doc_id": item["doc_id"],
                }
                points.append(self.get_point(embeddings, metadata))
            return points

        self._process_and_upload(
            collection_name=self.TEXT_COLLECTION,
            items=text_chunks,
            processor=batch_text_processor,
        )

    def _search_text_core(self, vectors: dict[str, Any], k: int):
        return self.client.query_points(
            collection_name=self.TEXT_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(**vectors["sparse"]),
                    using="sparse",
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=vectors["dense"],
                    using="dense",
                    limit=k * 2,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k
        )

    def _search_image_core(self, vectors: dict[str, Any], k: int):
        return self.client.query_points(
            collection_name=self.IMAGE_COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(**vectors["sparse"]),
                    using="sparse",
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=vectors["dense"],
                    using="dense",
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=vectors["image"],
                    using="image",
                    limit=k * 2,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k
        )

    def retrieve_all(self, query: str, k_text: int = 5, k_image: int = 3) -> dict[str, list[Document]]:
        """
        Calculates embeddings ONCE, then retrieves from BOTH collections.
        """
        vectors = self.embedding_service.embed_hybrid(query, mode=EmbeddingMode.QUERY)

        text_results = self._search_text_core(vectors, k_text)
        image_results = self._search_image_core(vectors, k_image)

        return {
            "texts": [
                Document(
                    page_content=p.payload.get("text", ""),
                    metadata={"doc_id": p.payload.get("doc_id"), "pages": p.payload.get("pages")}
                ) for p in text_results.points
            ],
            "images": [
                Document(
                    page_content=p.payload.get("caption", ""),
                    metadata={"path": p.payload.get("path"), "doc_id": p.payload.get("doc_id"), 'page': p.payload.get("page")}
                ) for p in image_results.points
            ]
        }

    @staticmethod
    def text_documents_to_llm_context(docs: list[Document]):
        chunks = []
        for doc in docs:
            chunks.append(
                '[ДОКУМЕНТ]\n'
                 f"Джерело: {get_textbook_source(doc.metadata['doc_id'])}\n"
                 f"Сторінки: {list_to_interval(doc.metadata['pages'])}\n"
                 f"Контекст: {doc.page_content}"
            )

        return '\n---\n'.join(chunks)

    @staticmethod
    def image_documents_to_llm_context(docs: list[Document]):
        chunks = []
        for doc in docs:
            chunks.append(
                '[ОПИС ЗОБРАЖЕННЯ]\n'
                 f"Джерело: {get_textbook_source(doc.metadata['doc_id'])}\n"
                 f"Сторінка: {list_to_interval(doc.metadata['page'])}\n"
                 f"Опис зображення: {doc.page_content}"
            )
        return '\n---\n'.join(chunks)

    @staticmethod
    def images_for_ui(docs: list[Document]) -> list[np.ndarray]:
        images = []
        for doc in docs:
            image = read_image(Path(doc.metadata["path"]))
            if image is not None:
                images.append(image)
        return images

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """
        Standard interface. Only fetches text by default.
        Uses the shared optimized pipeline internally.
        """
        return self.retrieve_all(query, k, **kwargs)["texts"]

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """
        Required by LangChain's VectorStore abstract class.
        We don't use it because we have a custom hybrid ingestion pipeline.
        """
        raise NotImplementedError("Use 'add_text_entry' or 'add_image_entry' instead.")

    def add_texts(self, texts, metadatas=None, **kwargs):
        """
        Required by LangChain's VectorStore abstract class.
        """
        raise NotImplementedError("Use 'add_text_entry' instead.")

    def run(self):
        logger.info("Starting indexing process...")

        if CHUNKS_DF_PATH.exists():
            try:
                logger.info(f"Loading chunks from {CHUNKS_DF_PATH}")
                chunks_df = read_parquet(CHUNKS_DF_PATH)

                text_chunks = chunks_df.to_dict(orient="records")
                self.add_text_entry(text_chunks)

                logger.info("Text indexing complete.")
            except Exception as error:
                logger.exception("Text indexing failed", exc_info=error)

        else:
            logger.warning(f"Chunks dataframe {CHUNKS_DF_PATH} does not exist. Skipping text indexing.")

        if IMAGES_DF_PATH.exists():
            try:
                logger.info(f"Loading images from {IMAGES_DF_PATH}")
                images_df = read_parquet(IMAGES_DF_PATH)

                images = images_df.to_dict(orient="records")
                self.add_image_entry(images)

                logger.info("Image indexing complete.")
            except Exception as error:
                logger.exception("Image indexing failed", exc_info=error)

        else:
            logger.warning(f"Images dataframe {IMAGES_DF_PATH} does not exist. Skipping image indexing.")

        logger.info("Finished indexing process.")
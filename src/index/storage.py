from typing import Any, Iterable, Callable
import pandas as pd
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models

import hashlib
import uuid
import json

from src.index.embedder import EmbeddingMode


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embedding_service,
        host: str = "localhost",
        port: int = 6333,
        client: QdrantClient | None = None):

        self.client = client or QdrantClient(host=host, port=port)
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
        vector = {name: embedding if 'sparse' not in name else models.SparseVector(
                    indices=embedding["indices"],
                    values=embedding["values"]
                ) for name, embedding in embeddings.items()}

        content_str = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
        hash_bytes = hashlib.sha256(content_str.encode('utf-8')).digest()

        point_id = str(uuid.UUID(bytes=hash_bytes[:16]))

        return models.PointStruct(
            id=point_id,
            vector=vector,
            payload=metadata
        )


    def _process_and_upload(
        self,
        collection_name: str,
        items: Iterable[Any],
        processor: Callable,
        batch_size: int = 64
    ):
        """
        Generic helper to loop through items, process them into embeddings/metadata,
        and upload to Qdrant.
        """
        batch: list = []

        for item in items:
            embeddings, metadata = processor(item)

            point = self.get_point(embeddings, metadata)
            batch.append(point)

            if len(batch) >= batch_size:
                self.client.upload_points(
                    collection_name=collection_name,
                    points=batch,
                    batch_size=batch_size,
                    wait=True
                )

                batch.clear()

        if batch:
            self.client.upload_points(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

    def add_image_entry(self, images_df: pd.DataFrame):
        def image_processor(row):
            return (
                self.embedding_service.embed_hybrid(text=row.caption, image=row.path, mode=EmbeddingMode.INDEX),
                {
                    "caption": row.caption,
                    "path": str(row.path),
                    "doc_id": row.doc_id,
                    "page": row.page,
                }
            )

        self._process_and_upload(
            collection_name=self.IMAGE_COLLECTION,
            items=images_df.itertuples(index=False),
            processor=image_processor
        )

    def add_text_entry(self, text_chunks: list[dict]):
        def text_processor(chunk):
            return (
                self.embedding_service.embed_text(chunk["text"], mode=EmbeddingMode.INDEX),
                {
                    "text": chunk["text"],
                    "pages": chunk["pages"],
                    "doc_id": chunk["doc_id"],
                }
            )

        self._process_and_upload(
            collection_name=self.TEXT_COLLECTION,
            items=text_chunks,
            processor=text_processor
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
                    query=vectors["sparse"],
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
                    metadata={"path": p.payload.get("path"), "doc_id": p.payload.get("doc_id")}
                ) for p in image_results.points
            ]
        }


    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """
        Standard interface. Only fetches text by default.
        Uses the shared optimized pipeline internally.
        """
        return self.retrieve_all(query, k, **kwargs)["texts"]

        # ... inside QdrantVectorStore class ...

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
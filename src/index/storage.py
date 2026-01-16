from typing import Callable, Iterable, Any
import pandas as pd
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient, models
import uuid


class QdrantVectorStore(VectorStore):
    def __init__(self, embedding_service):
        self.client = QdrantClient("localhost", port=6333)
        self.embedding_service = embedding_service

        self.TEXT_COLLECTION = "ukrainian_historical_text"
        self.IMAGE_COLLECTION = "ukrainian_historical_image"

        self._init_collections()

    def _init_collections(self):
        collections_config = {
            self.IMAGE_COLLECTION: {
                "vectors_config": {
                    'caption_dense_vector': models.VectorParams(
                        size=self.embedding_service.DENSE_DIM,
                        distance=models.Distance.COSINE
                    ),
                    'image_vector': models.VectorParams(
                        size=getattr(self.embedding_service, 'CLIP_DIM', 512),
                        distance=models.Distance.COSINE
                    ),
                },
                "sparse_vectors_config": {
                    'caption_sparse_vector': models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    ),
                }
            },
            self.TEXT_COLLECTION: {
                "vectors_config": {
                    'text_dense_vector': models.VectorParams(
                        size=self.embedding_service.DENSE_DIM,
                        distance=models.Distance.COSINE
                    ),
                },
                "sparse_vectors_config": {
                    'text_sparse_vector': models.SparseVectorParams(
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

        return models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=metadata
        )


    def _process_and_upload(self, collection_name: str, items: Iterable[Any], processor: Callable):
        """
        Generic helper to loop through items, process them into embeddings/metadata,
        and upload to Qdrant.
        """
        points = []

        for item in items:
            embeddings, metadata = processor(item)

            point = self.get_point(embeddings, metadata)
            points.append(point)

        self.client.upload_points(
            collection_name=collection_name,
            points=points,
            batch_size=64,
            wait=True
        )

    def add_image_entry(self, images_df: pd.DataFrame):
        def image_processor(row):
            return (
                self.embedding_service.embed_hybrid(text=row.caption, image=row.path),
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
                self.embedding_service.embed_text(chunk["text"]),
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

    def search_text_collection(self, query: str, top_k: int = 5):
        embeddings = self.embedding_service.embed_text(query)
        sparse_embedding = embeddings["text_sparse_vector"]
        dense_embedding = embeddings["text_dense_vector"]

        search_result = self.client.query_points(
            collection_name="ukrainian_historical_text",
            prefetch=[
                models.Prefetch(
                    query=sparse_embedding,
                    using="text_sparse_vector",
                    limit=20,
                ),
                models.Prefetch(
                    query=dense_embedding,
                    using="text_dense_vector",
                    limit=20,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
        )

        return search_result


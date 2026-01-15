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


    def add_image_entry(self, images_df: pd.DataFrame):
        points = []

        for image in images_df.itertuples(index=False):
            caption = image.caption
            image_path = image.path

            embeddings = self.embedding_service.embed_hybrid(
                text=caption,
                image=image_path
            )

            metadata = {
                "caption": caption,
                "path": str(image_path),
                "doc_id": image.doc_id,
                "page": image.page,
            }
            point = self.get_point(embeddings, metadata)
            points.append(point)

        self.client.upsert(
            collection_name=self.IMAGE_COLLECTION,
            points=points
        )

    def add_text_entry(self, text_chunks: list[dict]):
        points = []

        for chunk in text_chunks:
            text = chunk["text"]
            embeddings = self.embedding_service.embed_text(text)

            metadata = {
                "text": text,
                "pages": chunk["pages"],
                "doc_id": chunk["doc_id"],
            }
            point = self.get_point(embeddings, metadata)
            points.append(point)

        self.client.upsert(
            collection_name=self.TEXT_COLLECTION,
            points=points
        )
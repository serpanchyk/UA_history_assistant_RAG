from qdrant_client import models

from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore

embedder = EmbeddingService()
storage = QdrantVectorStore(embedder)

# 1. Dense only
def search_dense_text(query, k):
    vectors = embedder.embed_text(query)
    return storage.client.query_points(
        collection_name=storage.TEXT_COLLECTION,
        prefetch=[
            models.Prefetch(query=vectors['dense'], using='dense', limit=k)
        ],
        query=vectors['dense'],
        using='dense',
        limit=k,
        with_payload=True
    ).points

# 2. Hybrid (Dense + Sparse)
def search_hybrid_text(query, k):
    vectors = embedder.embed_hybrid(query)
    search_result = storage._search_text_core(vectors, k)
    return search_result.points

# 1. Clip
def search_clip_only(query, k):
    vector = embedder.embed_image(query)

    return storage.client.query_points(
        collection_name=storage.IMAGE_COLLECTION,
        query=vector,
        using='image',
        limit=k,
        with_payload=True
    ).points

# 2. CLIP + Dense
def search_clip_dense(query, k):
    vectors = embedder.embed_hybrid(query)
    # Ручний запит Qdrant без Sparse
    return storage.client.query_points(
        collection_name=storage.IMAGE_COLLECTION,
        prefetch=[
            models.Prefetch(query=vectors['dense'], using='dense', limit=k*2),
            models.Prefetch(query=vectors['image'], using='image', limit=k*2),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True
    ).points

# 3. Hybrid (CLIP + Dense + Sparse)
def search_hybrid_image(query, k):
    vectors = embedder.embed_hybrid(query)
    return storage._search_image_core(vectors, k).points
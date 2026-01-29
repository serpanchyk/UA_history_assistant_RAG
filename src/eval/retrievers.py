from qdrant_client import models

from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore

embedder = EmbeddingService()
storage = QdrantVectorStore(embedder)

# 1. Dense only
def search_dense_text(query, k):
    vectors = embedder.embed_text(query)
    return storage.client.search(
        collection_name=storage.TEXT_COLLECTION,
        query_vector=vectors['dense'],
        using='dense',
        limit=k,
        with_payload=True
    )

# 2. Hybrid (Dense + Sparse)
def search_hybrid_text(query, k):
    vectors = embedder.embed_hybrid(query)
    search_result = storage._search_text_core(vectors, k)
    return search_result.points


def search_clip_only(query, k):
    vectors = embedder.embed_image(query) # CLIP embedding
    return storage.client.search(
        collection_name=storage.IMAGE_COLLECTION,
        query_vector=vectors, # Це список float
        using='image',
        limit=k,
        with_payload=True
    )

# 2. CLIP + Dense (Картинка + Семантика опису)
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

# 3. Hybrid (CLIP + Dense + Sparse) - Ваш максимум
def search_full_hybrid(query, k):
    vectors = embedder.embed_hybrid(query)
    return storage._search_image_core(vectors, k).points
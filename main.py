from src.ingest.ingesting import PDFIngestor
from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore

if __name__ == "__main__":

    ingester = PDFIngestor()
    ingester.run()

    embedder = EmbeddingService()
    storage = QdrantVectorStore(embedder)

    storage.run()



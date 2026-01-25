from src.ingest.ingesting import PDFIngestor
from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore
from src.rag.llm_service import LLMService

if __name__ == "__main__":

    embedder = EmbeddingService()
    storage = QdrantVectorStore(embedder)

    llm_service = LLMService(storage)

    while True:
        query = input("Enter query: ")
        if query == 'q':
            break

        response = llm_service.generate_response(query)
        print(response)



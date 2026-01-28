import gradio as gr

from src import IMAGES_DIR_PATH
from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore
from src.rag.llm_service import LLMService

embedder = EmbeddingService()
storage = QdrantVectorStore(embedder)
llm_service = LLMService(storage)

def predict(message, history):
    """
    Core logic handler for the chat.
    Gradio handles the 'history' state automatically, so we don't need the custom ChatState class
    for basic UI retention, though you can still use it internally if it manages token limits/logic.
    """

    context = llm_service.retrieve(message)

    partial_response = ""
    generator = llm_service.generate_response(message, context)

    for chunk in generator:
        partial_response += chunk
        yield partial_response

    if context.images:
        partial_response += context.images
        yield partial_response


demo = gr.ChatInterface(
    fn=predict,
    title="ШІ-помічник з вивчення Історії України 🤖",
    examples=[
        "Поштовхом до розгортання в XVI ст. братського руху на українських землях було",
        "Собор святого Юри є прикладом архітектури"
    ],
    fill_height=True
)

if __name__ == "__main__":
    demo.launch(allowed_paths=[str(IMAGES_DIR_PATH)])


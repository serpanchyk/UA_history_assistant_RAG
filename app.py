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

    if hasattr(context, 'images') and context.images:
        for image_path in context.images:
            partial_response += f"\n\n![]({image_path})"
        yield partial_response


demo = gr.ChatInterface(
    fn=predict,
    title="ШІ-помічник з вивчення Історії України 🤖",
    examples=["Розкажи про Київську Русь", "Коли була прийнята незалежність?"]
)

if __name__ == "__main__":
    demo.launch(allowed_paths=[IMAGES_DIR_PATH])


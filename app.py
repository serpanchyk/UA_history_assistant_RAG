import streamlit as st

from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore
from src.rag.llm_service import LLMService
from src.ui.chat import ChatState

embedder = EmbeddingService()
storage = QdrantVectorStore(embedder)

llm_service = LLMService(storage)

st.set_page_config(
    page_title="UA History Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("ШІ-помічник з вивчення Історії України")

if "chat" not in st.session_state:
    st.session_state["chat"] = ChatState()

chat = st.session_state.chat

for msg in chat.messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

        for image in msg.images:
            st.image(image)

query = st.chat_input("Питайте, що цікавить")

if query:
    chat.add_user_message(query)

    context = llm_service.retrieve(query)

    with st.chat_message('assistant'):
        full_response = st.write_stream(
            llm_service.generate_response(query, context)
        )

        for image in context.images:
            st.image(image)

    chat.add_assistant_message(full_response, context.images)


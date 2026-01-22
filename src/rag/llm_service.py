from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from src.fs_io.filesystem import read_text_file
from src import PROMPTS_DIR_PATH
from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore


class LLMService:
    def __init__(self, api_key: str):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001"
        )

        self.embedding_service = EmbeddingService()

        self.storage = QdrantVectorStore(embedding_service=self.embedding_service)

        self.history = [{
            'role': 'system',
            'content': read_text_file(PROMPTS_DIR_PATH / "system_prompt.txt")
        }]

        self.chain = (
            RunnableLambda(self._build_messages)
            | self.model
            | StrOutputParser()
        )


    def _build_messages(self, user_input: str):
        prompt = PromptTemplate(
            input_variables=['text_context', 'image_context', 'query'],
            template = read_text_file(PROMPTS_DIR_PATH / "prompt_template.txt")
        )

        retrieved = self.storage.retrieve_all(user_input)




        prompt.format(query=user_input)



    def generate_response(self, user_input: str) -> str:






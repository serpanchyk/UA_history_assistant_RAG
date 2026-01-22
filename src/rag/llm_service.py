from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from src.fs_io.filesystem import read_text_file
from src import PROMPTS_DIR_PATH
from src.index.embedder import EmbeddingService
from src.index.storage import QdrantVectorStore


class LLMService:
    def __init__(self, storage: QdrantVectorStore):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001"
        )

        self.storage = storage

        self.rag_template_str = read_text_file(PROMPTS_DIR_PATH / "prompt_template.txt")

        self.system_message = SystemMessage(content=read_text_file(PROMPTS_DIR_PATH / "system_prompt.txt"))

        self.history = []
        self.MAX_HISTORY_LEN = 10

    def _condense_query(self, query: str) -> str:
        """If history exists, uses it to condence query to be self-sufficient"""

        condense_prompt = read_text_file(PROMPTS_DIR_PATH / 'condense_prompt.txt')

        history_text = '\n'.join([f'{msg.type}: {msg.content}' for msg in self.history])

        prompt = condense_prompt.format(chat_history=history_text, query=query)

        response = self.model.invoke([HumanMessage(content=prompt)])

        return response.content

    def _retrieve_context(self, query: str):
        """Retrieves data and formats it for the immediate turn only."""
        retrieved = self.storage.retrieve_all(query)

        return {
            'text_context': QdrantVectorStore.text_documents_to_llm_context(retrieved['texts']),
            'image_context': QdrantVectorStore.image_documents_to_llm_context(retrieved['images']),
            'query': query
        }

    def generate_response(self, query: str):

        condensed_query = self._condense_query(query)

        context = self._retrieve_context(condensed_query)

        formatted_prompt = self.prompt_template.format(
            text_context=context['text_context'],
            image_context=context['image_context'],
            query=condensed_query
        )

        messages = [self.system_message] + self.history + [HumanMessage(content=formatted_prompt)]

        response_content = self.model.invoke(messages).content

        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=response_content))

        if len(self.history) > self.MAX_HISTORY_LEN:
            self.history = self.history[-self.MAX_HISTORY_LEN:]

        return response_content
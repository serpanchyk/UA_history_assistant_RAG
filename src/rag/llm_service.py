from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

from typing import Any

from src.fs_io.filesystem import read_text_file
from src import PROMPTS_DIR_PATH
from src.index.storage import QdrantVectorStore
from src.utils.texts import chat_to_string


class LLMService:
    def __init__(self, storage: QdrantVectorStore):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001"
        )

        self.storage = storage

        self.rag_template = PromptTemplate.from_template(read_text_file(PROMPTS_DIR_PATH / "prompt_template.txt"))
        self.condense_template = PromptTemplate.from_template(read_text_file(PROMPTS_DIR_PATH / "condense_template.txt"))
        self.summarize_template = PromptTemplate.from_template(read_text_file(PROMPTS_DIR_PATH / "summarize_template.txt"))

        self.system_message = SystemMessage(content=read_text_file(PROMPTS_DIR_PATH / "system_prompt.txt"))

        self.summary = ''
        self.history = []
        self.MAX_HISTORY_LEN = 10

    def _update_summary(self):

        if len(self.history) <= self.MAX_HISTORY_LEN:
            return

        messages_to_summarize = self.history[:2]
        messages_to_keep = self.history[2:]

        conversation_text = chat_to_string(messages_to_summarize)


        prompt = self.summarize_template.format(current_summary = self.summary, new_messages=conversation_text)

        response = self.model.invoke([HumanMessage(content=prompt)])

        self.summary = response.content

        self.history = messages_to_keep

    def _condense_query(self, query: str) -> str:
        """If history exists, uses it to condence query to be self-sufficient"""

        recent_history = self.history[-4:]
        history_text = chat_to_string(recent_history)

        prompt = self.condense_template.format(chat_history=history_text, query=query)

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

        formatted_prompt = self.rag_template.format(
            text_context=context['text_context'],
            image_context=context['image_context'],
            query=condensed_query
        )

        messages: list[Any] = [self.system_message]

        if self.summary:
            summary_message = SystemMessage(content=f"Summary of past conversation: {self.summary}")
            messages.append(summary_message)

        messages.extend(self.history)
        messages.append(HumanMessage(content=formatted_prompt))

        response_content = self.model.invoke(messages).content

        self.history.append(HumanMessage(content=query))
        self.history.append(AIMessage(content=response_content))

        self._update_summary()

        return response_content
import unittest
from unittest.mock import patch

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

from src.rag.llm_service import LLMService


class TestLLMService(unittest.TestCase):
    def setUp(self):
        self.patches = {
            "model": patch("src.rag.llm_service.ChatGoogleGenerativeAI"),
            "storage": patch("src.rag.llm_service.QdrantVectorStore"),
        }

        self.mocks = {}
        for name, patcher in self.patches.items():
            mock = patcher.start()
            self.addCleanup(patcher.stop)
            self.mocks[name] = mock

        fake_message = unittest.mock.Mock()
        fake_message.content = "dummy response"

        self.mocks['model'].return_value.invoke.return_value = fake_message

        storage_instance = self.mocks["storage"].return_value
        self.llm_service = LLMService(storage=storage_instance)

    def test_init(self):

        self.assertEqual(self.llm_service.model, self.mocks["model"].return_value)
        self.assertEqual(self.llm_service.storage, self.mocks["storage"].return_value)
        self.assertEqual(self.llm_service.summary, '')
        self.assertEqual(self.llm_service.history, [])

        self.assertIsInstance(self.llm_service.rag_template, PromptTemplate)
        self.assertIsInstance(self.llm_service.condense_template, PromptTemplate)
        self.assertIsInstance(self.llm_service.summarize_template, PromptTemplate)
        self.assertIsInstance(self.llm_service.system_prompt, SystemMessage)

    @patch("src.rag.llm_service.chat_to_string")
    def test_condense_query_empty_history(self, mock_chat_to_string):
        mock_chat_to_string.return_value = ""

        self.llm_service.history = []

        response = self.llm_service._condense_query('dummy query')

        mock_chat_to_string.assert_called_once_with([])

        model = self.mocks["model"].return_value
        model.invoke.assert_called_once()

        messages = model.invoke.call_args[0][0]
        self.assertEqual(len(messages), 1)
        prompt = messages[0].content

        self.assertIn("dummy query", prompt)

        self.assertEqual(response, "dummy response")

    @patch("src.rag.llm_service.chat_to_string")
    def test_condense_query_correct_history(self, mock_chat_to_string):
        mock_chat_to_string.return_value = "8, 9"
        self.llm_service.history = [
            HumanMessage("1"),
            AIMessage("2"),
            HumanMessage("3"),
            AIMessage("4"),
            HumanMessage("5"),
            AIMessage("6"),
            HumanMessage("7"),
            AIMessage("8"),
            HumanMessage("9"),
        ]

        self.llm_service.SHORT_MEMORY = 2

        response = self.llm_service._condense_query('dummy query')

        mock_chat_to_string.assert_called_once_with(
            self.llm_service.history[-2:]
        )

        model = self.mocks["model"].return_value
        model.invoke.assert_called_once()

        messages = model.invoke.call_args[0][0]
        self.assertEqual(len(messages), 1)
        prompt = messages[0].content

        self.assertIn("dummy query", prompt)
        self.assertIn('8, 9', prompt)
        self.assertIn('dummy query', prompt)

        self.assertEqual(response, "dummy response")

    @patch("src.rag.llm_service.chat_to_string")
    def test_update_summary_short_history(self, mock_chat_to_string):

        self.llm_service.MAX_HISTORY_LEN = 5
        self.llm_service.history = [HumanMessage("1")]

        self.llm_service._update_summary()

        mock_chat_to_string.assert_not_called()

        model = self.mocks["model"].return_value
        model.invoke.assert_not_called()

        self.assertEqual(self.llm_service.summary, '')
        self.assertEqual(self.llm_service.history,  [HumanMessage("1")])

    def test_update_summary_long_history(self):
        self.llm_service.MAX_HISTORY_LEN = 5
        self.llm_service.history = [
            HumanMessage("Old human message"),
            AIMessage("Old ai message"),
            HumanMessage("3"),
            AIMessage("4"),
            HumanMessage("5"),
            AIMessage("6"),
            HumanMessage("7"),
        ]
        self.llm_service.summary = 'Old Summary'

        self.llm_service._update_summary()

        model = self.mocks["model"].return_value
        model.invoke.assert_called_once()

        messages = model.invoke.call_args[0][0]
        self.assertEqual(len(messages), 1)
        prompt = messages[0].content

        self.assertIn("Old Summary", prompt)
        self.assertIn('Old human message', prompt)
        self.assertIn('Old ai message', prompt)

        self.assertEqual(self.llm_service.summary, 'dummy response')
        self.assertEqual(self.llm_service.history,
    [HumanMessage("3"),
            AIMessage("4"),
            HumanMessage("5"),
            AIMessage("6"),
            HumanMessage("7"),])

    def test_generate_response_without_summary(self):
        """
        Should build prompt without summary system message when summary is empty,
        invoke the model, update history correctly and return model response.
        """
        self.llm_service.history = [
            HumanMessage("hi"),
            AIMessage("hello"),
        ]

        response = self.llm_service.generate_response("new question")

        model = self.mocks["model"].return_value
        model.invoke.assert_called()

        messages = model.invoke.call_args[0][0]

        self.assertIsInstance(messages[0], SystemMessage)
        self.assertNotIn("Короткий виклад", messages[0].content)

        self.assertEqual(self.llm_service.history[-2].content, "new question")
        self.assertEqual(self.llm_service.history[-1].content, "dummy response")

        self.assertEqual(response, "dummy response")

    def test_generate_response_with_summary(self):
        """
        Should include summary system message when summary exists
        and place it before raw history.
        """
        self.llm_service.summary = "Previous summary"
        self.llm_service.history = [
            HumanMessage("hi"),
            AIMessage("hello"),
        ]

        self.llm_service.generate_response("question")

        model = self.mocks["model"].return_value
        messages = model.invoke.call_args[0][0]

        self.assertIsInstance(messages[0], SystemMessage)

        self.assertIsInstance(messages[1], SystemMessage)
        self.assertIn("Previous summary", messages[1].content)

        self.assertEqual(messages[2].content, "hi")
        self.assertEqual(messages[3].content, "hello")

    def test_generate_response_calls_internal_pipeline(self):
        """
        Should call condense, retrieve and update_summary exactly once
        during response generation.
        """
        with patch.object(self.llm_service, "_condense_query") as mock_condense, \
                patch.object(self.llm_service, "_retrieve_context") as mock_retrieve, \
                patch.object(self.llm_service, "_update_summary") as mock_update:
            mock_condense.return_value = "condensed"
            mock_retrieve.return_value = {
                "text_context": "",
                "image_context": "",
                "query": "condensed",
            }

            self.llm_service.generate_response("question")

            mock_condense.assert_called_once_with("question")
            mock_retrieve.assert_called_once_with("condensed")
            mock_update.assert_called_once()

    def test_generate_response_history_grows_by_two(self):
        """
        Each generate_response call should append exactly
        one HumanMessage and one AIMessage to history.
        """
        initial_len = len(self.llm_service.history)

        self.llm_service.generate_response("question")

        self.assertEqual(len(self.llm_service.history), initial_len + 2)

    def test_generate_response_history_is_even_length(self):
        """
        History length must always remain even
        (human + ai message pairs).
        """
        for _ in range(3):
            self.llm_service.generate_response("question")

        self.assertEqual(len(self.llm_service.history) % 2, 0)

    def test_summary_not_added_to_history(self):
        """
        Summary must never be persisted inside history,
        only injected into prompt as a SystemMessage.
        """
        self.llm_service.summary = "Some summary"
        self.llm_service.generate_response("question")

        for msg in self.llm_service.history:
            self.assertNotIn("Some summary", msg.content)


if __name__ == "__main__":
    unittest.main()



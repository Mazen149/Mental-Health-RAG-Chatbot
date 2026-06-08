import os
import unittest
import pickle
import tempfile
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.documents import Document

# Assuming your main class is in a file named rag_pipeline.py inside the parent directory
# We adjust the import paths or use mocking safely
from src.modules.rag import MentalHealthRAG

class TestMentalHealthRAG(unittest.TestCase):

    @patch('src.modules.rag.QdrantClient')
    @patch('src.modules.rag.HuggingFaceEmbeddings')
    @patch('src.modules.rag.CrossEncoder')
    @patch('src.modules.rag.Groq')
    def setUp(self, mock_groq, mock_cross, mock_hf, mock_qdrant):
        """Set up a fresh instance of the RAG class with mocked heavy models."""
        self.cache_path = os.path.join(
            tempfile.gettempdir(), f"test_cache_{os.getpid()}_{self._testMethodName}.pkl"
        )
        self.rag = MentalHealthRAG(qdrant_path="./test_db", cache_path=self.cache_path)

    def tearDown(self):
        """Clean up test artifacts after tests complete."""
        if hasattr(self, "rag"):
            self.rag.close()
        if os.path.exists(self.cache_path):
            try:
                os.remove(self.cache_path)
            except PermissionError:
                pass

    @patch('pandas.read_json')
    def test_load_and_preprocess_generates_documents(self, mock_read_json):
        """Test that raw JSON is properly transformed into LangChain Documents."""
        # Mocking incoming pandas DataFrame structure
        mock_data = pd.DataFrame([
            {"Context": "I feel anxious.", "Response": "Take a deep breath."},
            {"Context": "I feel anxious.", "Response": "Try to meditate."},  # Duplicate context to check grouping
            {"Context": "Context0", "Response": "Ignore row 0"} # Simulating row 0 drop logic
        ])
        mock_read_json.return_value = mock_data

        documents = self.rag.load_and_preprocess(dataset_url="dummy_url")
        
        # Verify document structural conversion
        self.assertIsInstance(documents, list)
        self.assertEqual(len(documents), 1) # Context0 dropped, duplicates merged
        self.assertEqual(documents[0].page_content, "I feel anxious.\n\nTake a deep breath. Try to meditate.")
        self.assertEqual(documents[0].metadata["question"], "I feel anxious.")
        self.assertIn("Take a deep breath.", documents[0].metadata["response"])

    @patch('pandas.read_json')
    def test_caching_mechanism(self, mock_read_json):
        """Test that the pipeline saves to pickle cache and reads from it correctly."""
        mock_data = pd.DataFrame([
            {"Context": "Context0", "Response": "Row 0"},
            {"Context": "Test Query", "Response": "Test Answer"}
        ])
        mock_read_json.return_value = mock_data

        # 1. First run: Generates cache file
        self.assertFalse(os.path.exists(self.cache_path))
        first_run_docs = self.rag.load_and_preprocess("dummy_url")
        self.assertTrue(os.path.exists(self.cache_path))

        # 2. Second run: Should read directly from cache without calling read_json again
        mock_read_json.reset_mock()
        second_run_docs = self.rag.load_and_preprocess("dummy_url")
        
        mock_read_json.assert_not_called()
        self.assertEqual(len(first_run_docs), len(second_run_docs))

    def test_query_aborts_without_setup(self):
        """Ensure query method handles uninitialized retriever gracefully."""
        response = self.rag.query("Hello")
        self.assertEqual(response["answer"], "Retriever not set up.")
        self.assertEqual(response["resources"], [])

    @patch('src.modules.rag.EnsembleRetriever')
    def test_language_alignment_prompting(self, mock_ensemble):
        """Verify LLM payload includes query context properly during execution."""
        # Setup mock hybrid retriever output
        mock_doc = Document(page_content="Mock question", metadata={"response": "Mock answer in English"})
        self.rag.ensemble_retriever = MagicMock()
        self.rag.ensemble_retriever.invoke.return_value = [mock_doc]
        
        # Mock Reranker scores
        self.rag.rerank_model.predict = MagicMock(return_value=[0.99])

        # Mock Groq LLM API response
        mock_choice = MagicMock()
        mock_choice.message.content = "Respuesta simulada en Español"
        self.rag.client.chat.completions.create.return_value.choices = [mock_choice]

        # Act
        spanish_query = "¿Cómo manejar el estrés?"
        output = self.rag.query(spanish_query)

        # Assert LLM was called and output received
        self.assertEqual(output["answer"], "Respuesta simulada en Español")
        self.assertEqual(len(output["resources"]), 1)
        self.assertEqual(output["resources"][0]["response"], "Mock answer in English")
        self.rag.client.chat.completions.create.assert_called_once()

    def test_query_general_calls_llm(self):
        """Verify query_general correctly formats general conversation prompts and returns LLM text."""
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello Mazen! I'm here to support you."
        self.rag.client.chat.completions.create.return_value.choices = [mock_choice]
        
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        answer = self.rag.query_general("my name is mazen", history=history, language="English")
        
        self.assertEqual(answer, "Hello Mazen! I'm here to support you.")
        self.rag.client.chat.completions.create.assert_called_once()

    def test_query_condensation_with_history(self):
        """Verify that when history is provided, query condenses the follow-up query using LLM."""
        # Mock hybrid retriever output
        mock_doc = Document(page_content="Mock question", metadata={"response": "Mock answer"})
        self.rag.ensemble_retriever = MagicMock()
        self.rag.ensemble_retriever.invoke.return_value = [mock_doc]
        self.rag.rerank_model.predict = MagicMock(return_value=[0.99])

        # Mock Groq LLM API responses: first for query condensation, second for final generation
        mock_condense_choice = MagicMock()
        mock_condense_choice.message.content = "Standalone condensed query"
        
        mock_gen_choice = MagicMock()
        mock_gen_choice.message.content = "Grounded response content"
        
        mock_completions = MagicMock()
        mock_completions.choices = [mock_condense_choice]
        
        mock_completions_gen = MagicMock()
        mock_completions_gen.choices = [mock_gen_choice]
        
        # Side effect to return condensation first, then generation
        self.rag.client.chat.completions.create.side_effect = [mock_completions, mock_completions_gen]

        history = [{"role": "user", "content": "I am sad because my friends hate me."}]
        output = self.rag.query("why am i depressed", history=history)

        self.assertEqual(output["answer"], "Grounded response content")
        self.assertEqual(self.rag.client.chat.completions.create.call_count, 2)

if __name__ == '__main__':
    unittest.main()

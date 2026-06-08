import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.router import route_query, get_direct_greeting
from src.modules.rag import build_system_prompt, detect_crisis

class TestRouter(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.mock_rag = MagicMock()
        # Mock embeddings
        self.mock_rag.embeddings = MagicMock()
        
    async def test_direct_greeting_short_input(self):
        """Test that short greeting queries return direct greeting response immediately."""
        queries = ["hi", "hello", "مرحبا", "bonjour", "hola", "hey there"]
        for q in queries:
            result = await route_query(q, self.mock_rag)
            self.assertEqual(result["intent"], "greeting")
            self.assertEqual(result["resources"], [])
            self.assertIsNone(result["language"])
            self.assertIsNone(result["emotion"])
            self.assertTrue(len(result["answer"]) > 0)

    async def test_greeting_correct_language_resolution(self):
        """Test that short greetings in different languages resolve to the correct language greeting response."""
        test_cases = [
            ("hola", "¡Hola! Estoy aquí para apoyarte."),  # Spanish
            ("مرحبا", "مرحباً! أنا هنا لدعمك ومساعدتك."),    # Arabic
            ("hi", "Hello! I am here to support you."),     # English
            ("hallo", "Hallo! Ich bin hier, um Sie zu unterstützen.") # German
        ]
        for query, expected_start in test_cases:
            result = await route_query(query, self.mock_rag)
            self.assertEqual(result["intent"], "greeting")
            self.assertTrue(result["answer"].startswith(expected_start), f"Failed for query '{query}': got '{result['answer']}'")

    async def test_goodbye_correct_language_resolution(self):
        """Test that short goodbyes in different languages resolve to the correct language goodbye response."""
        test_cases = [
            ("adios", "¡Adiós! Cuídate mucho."), # Spanish
            ("مع السلامة", "مع السلامة! أتمنى لك يوماً هادئاً"), # Arabic
            ("bye", "Goodbye! Take care of yourself.") # English
        ]
        for query, expected_start in test_cases:
            result = await route_query(query, self.mock_rag)
            self.assertEqual(result["intent"], "goodbye")
            self.assertTrue(result["answer"].startswith(expected_start), f"Failed for query '{query}': got '{result['answer']}'")

    async def test_gratitude_correct_language_resolution(self):
        """Test that short gratitude messages in different languages resolve to the correct language gratitude response."""
        test_cases = [
            ("gracias", "¡De nada! Me alegra haber podido ayudarte."), # Spanish
            ("شكرا", "العفو! يسعدني أنني استطعت مساعدتك."), # Arabic
            ("thank you", "You're welcome! I'm glad I could help.") # English
        ]
        for query, expected_start in test_cases:
            result = await route_query(query, self.mock_rag)
            self.assertEqual(result["intent"], "gratitude")
            self.assertTrue(result["answer"].startswith(expected_start), f"Failed for query '{query}': got '{result['answer']}'")


    @patch('src.router.detect_language')
    @patch('src.router.classify_emotion')
    @patch('src.router.classify_intent')
    async def test_mental_health_routing_pipeline(self, mock_intent, mock_emotion, mock_lang):
        """Test that mental health related queries run the full classification and RAG pipeline."""
        mock_lang.return_value = "Arabic"
        mock_emotion.return_value = ["Sadness", "Fear"]
        mock_intent.return_value = "asking_mental_health_question"
        
        # Mock RAG query
        self.mock_rag.query.return_value = {
            "answer": "أنا أفهم شعورك بالخوف والحزن.",
            "resources": [{"score": 0.9, "page_content": "dummy", "response": "counselor response"}]
        }
        
        query = "أشعر بحزن شديد وخوف من المستقبل"
        result = await route_query(query, self.mock_rag)
        
        self.assertEqual(result["language"], "Arabic")
        self.assertEqual(result["emotion"], ["Sadness", "Fear"])
        self.assertEqual(result["intent"], "asking_mental_health_question")
        self.assertEqual(result["answer"], "أنا أفهم شعورك بالخوف والحزن.")
        self.assertEqual(len(result["resources"]), 1)
        self.mock_rag.query.assert_called_once()

    @patch('src.router.detect_language')
    @patch('src.router.classify_emotion')
    @patch('src.router.classify_intent')
    async def test_short_non_greeting_padding(self, mock_intent, mock_emotion, mock_lang):
        """Verify that short non-greeting queries are padded before language detection."""
        mock_lang.return_value = "English"
        mock_emotion.return_value = ["Sadness"]
        mock_intent.return_value = "asking_mental_health_question"
        
        self.mock_rag.query.return_value = {
            "answer": "Supportive response",
            "resources": []
        }
        
        # Test case 1: small query (1 word) -> should be padded with /p word tokens
        query_short = "sad"
        await route_query(query_short, self.mock_rag)
        mock_lang.assert_called_with("/p /p /p /p sad /p /p /p /p")
        
        # Test case 2: long query (6 words) -> should NOT be padded
        mock_lang.reset_mock()
        query_long = "I feel extremely sad and hopeless"
        await route_query(query_long, self.mock_rag)
        mock_lang.assert_called_with(query_long)

    def test_crisis_detection(self):
        """Verify that crisis keyword detection correctly identifies crisis queries."""
        self.assertTrue(detect_crisis("I want to commit suicide"))
        self.assertTrue(detect_crisis("أريد الانتحار"))
        self.assertTrue(detect_crisis("I want to end my life"))
        self.assertFalse(detect_crisis("I am feeling a bit stressed from work"))

    def test_build_system_prompt_has_tone_directives(self):
        """Verify that system prompt includes custom tone directives based on emotion list."""
        prompt = build_system_prompt(["Sadness", "Anger"], "English", "I am feeling down")
        self.assertIn("For Sadness: Validate their pain", prompt)
        self.assertIn("For Anger: Remain completely calm", prompt)
        self.assertNotIn("For Fear: Focus on safety", prompt)
        self.assertIn("English", prompt)

    def test_build_system_prompt_includes_crisis_helpline(self):
        """Verify that helpline message is appended to the system prompt in crisis cases."""
        prompt = build_system_prompt(["Fear"], "English", "I want to kill myself")
        self.assertIn("Suicide & Crisis Lifeline at 988", prompt)

        prompt_ar = build_system_prompt(["Fear"], "Arabic", "أريد الانتحار")
        self.assertIn("الاتصال بخط المساعدة الوطني للسلامة النفسية", prompt_ar)

    @patch('src.router.detect_language')
    @patch('src.router.classify_emotion')
    @patch('src.router.classify_intent')
    async def test_out_of_scope_routing(self, mock_intent, mock_emotion, mock_lang):
        """Test that out_of_scope queries get politely redirected in user's language."""
        mock_lang.return_value = "French"
        mock_emotion.return_value = []
        mock_intent.return_value = "out_of_scope"
        
        query = "Comment cuisiner des pâtes ?"
        result = await route_query(query, self.mock_rag)
        
        self.assertEqual(result["intent"], "out_of_scope")
        self.assertEqual(result["language"], "French")
        self.assertIn("santé mentale", result["answer"]) # French redirect message

    @patch('src.router.detect_language')
    @patch('src.router.classify_emotion')
    @patch('src.router.classify_intent')
    async def test_conversational_routing_pipeline(self, mock_intent, mock_emotion, mock_lang):
        """Test that conversational queries (greetings in Layer 2) use query_general on RAG."""
        mock_lang.return_value = "English"
        mock_emotion.return_value = []
        mock_intent.return_value = "greeting"
        
        self.mock_rag.query_general.return_value = "Hello! Nice to meet you."
        
        query = "hello my name is mazen"
        result = await route_query(query, self.mock_rag)
        
        self.assertEqual(result["intent"], "greeting")
        self.assertEqual(result["answer"], "Hello! Nice to meet you.")
        self.assertEqual(result["resources"], [])
        self.mock_rag.query_general.assert_called_once_with(
            user_query=query,
            history=None,
            language="English"
        )

if __name__ == '__main__':
    unittest.main()

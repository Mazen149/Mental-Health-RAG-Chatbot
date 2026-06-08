import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.modules.intent_classifier import IntentClassifier, IntentRouter, Intent

@pytest.fixture(autouse=True)
def mock_hf_inference():
    """Autouse fixture to mock Hugging Face InferenceClient during test execution."""
    with patch("src.modules.intent_classifier.InferenceClient") as mock_client_class:
        mock_client_inst = MagicMock()
        
        # Mock feature extraction to return dummy vectors of length 384
        def mock_extract(texts, model=None):
            if isinstance(texts, str):
                # Return single embedding vector
                return [0.05] * 384
            # Return list of embedding vectors
            return [[0.05] * 384 for _ in texts]
            
        mock_client_inst.feature_extraction.side_effect = mock_extract
        mock_client_class.return_value = mock_client_inst
        yield mock_client_inst

@pytest.fixture(autouse=True)
def mock_sentence_transformer():
    """Autouse fixture to mock SentenceTransformer during test execution."""
    with patch("sentence_transformers.SentenceTransformer") as mock_transformer_class:
        mock_transformer_inst = MagicMock()
        
        # Mock encode to return dummy vectors of length 384
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                return np.asarray([0.05] * 384, dtype=np.float32)
            return np.asarray([[0.05] * 384 for _ in texts], dtype=np.float32)
            
        mock_transformer_inst.encode.side_effect = mock_encode
        mock_transformer_class.return_value = mock_transformer_inst
        yield mock_transformer_inst

def test_intent_classifier_embedding_match():
    """Verify that classifier returns Intent object on successful embedding lookup."""
    classifier = IntentClassifier()
    
    # We force self._get_embedding to return a vector that aligns with general/mh/out_of_scope
    # Or simply mock the embedding classifier output
    with patch.object(classifier, "_embedding_classifier") as mock_embed:
        mock_embed.return_value = Intent(type="asking_mental_health_question", confidence=0.88, classifier="embedding")
        
        res = classifier.classify("I feel anxious")
        assert res.type == "asking_mental_health_question"
        assert res.confidence == 0.88
        assert res.classifier == "embedding"

def test_intent_classifier_llm_fallback_success():
    """Verify cascading to Groq fallback when embedding classifier returns None."""
    classifier = IntentClassifier()
    
    with patch.object(classifier, "_embedding_classifier", return_value=None):
        # Mock groq_client
        mock_groq = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '{"type": "out_of_scope", "confidence": 0.95}'
        mock_groq.chat.completions.create.return_value.choices = [mock_choice]
        classifier.groq_client = mock_groq
        
        res = classifier.classify("What is Python?")
        assert res.type == "out_of_scope"
        assert res.confidence == 0.95
        assert res.classifier == "llm"

def test_intent_classifier_default_fallback():
    """Verify default fallback triggers when Groq is not configured or fails."""
    classifier = IntentClassifier()
    classifier.groq_client = None
    
    with patch.object(classifier, "_embedding_classifier", return_value=None):
        res = classifier.classify("What is Python?")
        assert res.type == "greeting"
        assert res.confidence == 0.5
        assert res.classifier == "llm"

def test_intent_router():
    """Verify IntentRouter routes general, out_of_scope, and mental health questions correctly."""
    router = IntentRouter()
    
    with patch.object(router.classifier, "classify") as mock_classify:
        # 1. Greeting intent routing
        mock_classify.return_value = Intent(type="greeting", confidence=0.9, classifier="embedding")
        res = router.route("Hello!", detected_language="en")
        assert res["intent"] == "greeting"
        assert res["action"] == "direct_reply"
        assert res["use_rag"] is False
        assert "support you" in res["response"]
        
        # 2. Mental health question routing
        mock_classify.return_value = Intent(type="asking_mental_health_question", confidence=0.85, classifier="embedding")
        res_mh = router.route("I'm sad", detected_language="en")
        assert res_mh["intent"] == "asking_mental_health_question"
        assert res_mh["action"] == "rag_pipeline"
        assert res_mh["use_rag"] is True
        assert res_mh["response"] is None
        
        # 3. Out of scope routing
        mock_classify.return_value = Intent(type="out_of_scope", confidence=0.95, classifier="llm")
        res_oos = router.route("weather", detected_language="en")
        assert res_oos["intent"] == "out_of_scope"
        assert res_oos["action"] == "decline"
        assert res_oos["use_rag"] is False
        assert "specialised in mental health" in res_oos["response"]

def test_convenience_function():
    """Verify singleton convenience classifier wrapper."""
    from src.modules.intent_classifier import classify_intent
    
    with patch("src.modules.intent_classifier._classifier_instance") as mock_inst:
        mock_inst.classify.return_value = Intent(type="greeting", confidence=0.9, classifier="embedding")
        res = classify_intent("hi")
        assert res.type == "greeting"

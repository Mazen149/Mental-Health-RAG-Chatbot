import pytest
from unittest.mock import MagicMock
from chatbot.nlp.intent_classifier import IntentClassifier, IntentRouter, classify_intent

def test_intent_classifier_rule_based():
    """Verify Stage 1 Rule-Based classifier triggers on keyword patterns and false-positive guards."""
    classifier = IntentClassifier()
    
    # Test greeting rule-based trigger
    res = classifier.classify("Hi there!")
    assert res["intent"] == "greeting"
    assert res["stage"] == "rule_based"
    assert res["confidence"] == 0.95
    
    # Test gratitude rule-based trigger
    res = classifier.classify("thank you so much")
    assert res["intent"] == "gratitude"
    assert res["stage"] == "rule_based"
    
    # Test goodbye rule-based trigger
    res = classifier.classify("bye bye!")
    assert res["intent"] == "goodbye"
    assert res["stage"] == "rule_based"

def test_intent_classifier_embedding():
    """Verify Stage 2 Embedding classification matches semantic mental health queries to centroids."""
    classifier = IntentClassifier()
    
    # Test mental health questions trigger embedding centroid routing
    res = classifier.classify("I have been feeling extremely stressed and anxious lately")
    assert res["intent"] == "asking_mental_health_question"
    assert res["stage"] == "embedding"
    assert res["confidence"] >= 0.7

def test_intent_classifier_llm_fallback():
    """Verify Stage 3 LLM fallback is triggered and mock the Groq completions call for offline stability."""
    classifier = IntentClassifier(groq_api_key="mock_key")
    
    # Mock the Groq completions.create to return out_of_scope
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"intent": "out_of_scope", "confidence": 0.9}'))
    ]
    classifier._groq_client.chat.completions.create = MagicMock(return_value=mock_response)
    
    # A query that fails both Stage 1 and Stage 2 -> falls back to mocked Stage 3
    res = classifier.classify("What is the stock price of Tesla?")
    assert res["intent"] == "out_of_scope"
    assert res["stage"] == "llm"
    assert res["confidence"] == 0.9
    classifier._groq_client.chat.completions.create.assert_called_once()

def test_intent_router():
    """Verify that IntentRouter maps the classifier outputs to the RAG routing rules correctly."""
    router = IntentRouter()
    
    res = router.route("Hello!", detected_language="en")
    assert res["intent"] == "greeting"
    assert res["action"] == "direct_reply"
    assert res["use_rag"] is False
    assert res["response"] is not None

def test_convenience_function():
    """Verify module-level singleton convenience classifier."""
    res = classify_intent("hi")
    assert res["intent"] == "greeting"

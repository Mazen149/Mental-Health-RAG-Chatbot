import pytest
from src.modules.language_detector import LanguageDetector, detect_language, preprocess

def test_preprocess():
    """Verify that preprocessing cleans text, normalises unicode and collapses spaces correctly."""
    assert preprocess("HELLO WORLD 123!!!") == "hello world"
    assert preprocess("Visit https://google.com for info") == "visit for info"
    assert preprocess("Multiple   spaces    here") == "multiple spaces here"

def test_detector_basic_en():
    """Test standard English language detection."""
    detector = LanguageDetector()
    res = detector.detect("Hello, how are you today?")
    assert res["language"] == "en"
    assert res["language_name"] == "English"
    assert res["confidence"] > 0.4

def test_detector_arabic():
    """Test Arabic language detection with high confidence."""
    detector = LanguageDetector()
    res = detector.detect("مرحبا، كيف يمكنني مساعدتك اليوم؟")
    assert res["language"] == "ar"
    assert res["language_name"] == "Arabic"
    assert res["confidence"] > 0.8

def test_detect_top_k():
    """Test top-k language predictions and structures."""
    detector = LanguageDetector()
    res = detector.detect_top_k("Hello, how are you?", k=3)
    assert len(res) == 3
    assert res[0]["language"] == "en"
    assert "language_name" in res[0]
    assert "confidence" in res[0]

def test_convenience_function():
    """Verify the module-level singleton convenience function works correctly."""
    res = detect_language("Bonjour, comment allez-vous?")
    assert res["language"] == "fr"
    assert res["language_name"] == "French"

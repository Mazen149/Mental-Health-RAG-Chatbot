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

def test_short_sentences_fallback():
    """Verify high accuracy on 56 short user queries across multiple languages using FastText fallback."""
    TEST_CASES = [
        # --- English ---
        {"text": "i am sad", "expected": "English"},
        {"text": "help me please", "expected": "English"},
        {"text": "i feel nervous", "expected": "English"},
        {"text": "so lonely today", "expected": "English"},
        {"text": "very stressed out", "expected": "English"},
        {"text": "cannot sleep well", "expected": "English"},
        {"text": "anxiety is bad", "expected": "English"},
        {"text": "need to talk", "expected": "English"},

        # --- Arabic ---
        {"text": "أنا حزين جدا", "expected": "Arabic"},
        {"text": "ساعدني أرجوك", "expected": "Arabic"},
        {"text": "أشعر بقلق", "expected": "Arabic"},
        {"text": "أنا وحيد هنا", "expected": "Arabic"},
        {"text": "ضغوطات كبيرة", "expected": "Arabic"},
        {"text": "لا أستطيع النوم", "expected": "Arabic"},
        {"text": "أحتاج مساعدة", "expected": "Arabic"},
        {"text": "أشعر بالخوف", "expected": "Arabic"},

        # --- Spanish ---
        {"text": "estoy muy triste", "expected": "Spanish"},
        {"text": "ayúdame por favor", "expected": "Spanish"},
        {"text": "tengo mucho miedo", "expected": "Spanish"},
        {"text": "me siento solo", "expected": "Spanish"},
        {"text": "estoy muy estresado", "expected": "Spanish"},
        {"text": "necesito apoyo", "expected": "Spanish"},
        {"text": "no puedo dormir", "expected": "Spanish"},
        {"text": "siento mucha ansiedad", "expected": "Spanish"},

        # --- French ---
        {"text": "je suis triste", "expected": "French"},
        {"text": "aidez-moi s'il vous plaît", "expected": "French"},
        {"text": "j'ai très peur", "expected": "French"},
        {"text": "je me sens seul", "expected": "French"},
        {"text": "très stressé", "expected": "French"},
        {"text": "besoin d'aide", "expected": "French"},
        {"text": "insomnie et angoisse", "expected": "French"},
        {"text": "je ne vais pas bien", "expected": "French"},

        # --- German ---
        {"text": "ich bin traurig", "expected": "German"},
        {"text": "bitte hilf mir", "expected": "German"},
        {"text": "ich habe angst", "expected": "German"},
        {"text": "ich bin einsam", "expected": "German"},
        {"text": "sehr gestresst", "expected": "German"},
        {"text": "brauche hilfe", "expected": "German"},
        {"text": "kann nicht schlafen", "expected": "German"},
        {"text": "angstzustände", "expected": "German"},

        # --- Swahili ---
        {"text": "nahisi huzuni", "expected": "Swahili"},
        {"text": "nisaidie tafadhali", "expected": "Swahili"},
        {"text": "ninaogopa sana", "expected": "Swahili"},
        {"text": "nahisi upweke", "expected": "Swahili"},
        {"text": "nimefadhaika", "expected": "Swahili"},
        {"text": "nahitaji msaada", "expected": "Swahili"},
        {"text": "siwezi kulala", "expected": "Swahili"},
        {"text": "nina wasiwasi", "expected": "Swahili"},

        # --- Turkish ---
        {"text": "çok üzgünüm", "expected": "Turkish"},
        {"text": "lütfen yardım et", "expected": "Turkish"},
        {"text": "korkuyorum", "expected": "Turkish"},
        {"text": "kendimi yalnız hissediyorum", "expected": "Turkish"},
        {"text": "çok stresliyim", "expected": "Turkish"},
        {"text": "yardıma ihtiyacım var", "expected": "Turkish"},
        {"text": "uyuyamıyorum", "expected": "Turkish"},
        {"text": "endişeliyim", "expected": "Turkish"},
    ]
    
    correct = 0
    failures = []
    for case in TEST_CASES:
        res = detect_language(case["text"])
        if res["language_name"] == case["expected"]:
            correct += 1
        else:
            failures.append((case["text"], case["expected"], res["language_name"]))
            
    assert correct >= 54, f"Accuracy too low: {correct}/{len(TEST_CASES)} passed. Failures: {failures}"

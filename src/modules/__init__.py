"""
Mental Health RAG Chatbot Core Package.
Exposes key modules and convenience wrappers with signatures compatible with router.py.
"""

from .language_detector import LanguageDetector
from .emotion_classifier import EmotionClassifier

# Lazy singletons for the convenience functions
_language_detector = None
_emotion_classifier = None
_intent_classifier = None


def __getattr__(name: str):
    if name in {"IntentClassifier", "IntentRouter"}:
        from .intent_classifier import IntentClassifier, IntentRouter

        return IntentClassifier if name == "IntentClassifier" else IntentRouter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def detect_language(text: str) -> str:
    """
    Detects language and returns the full name of the language (e.g. 'English').
    Compatible with router.py.
    """
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    res = _language_detector.detect(text)
    return res.get("language_name", "English")


def classify_emotion(text: str) -> list[str]:
    """
    Classifies emotion and returns a list of 1 or 2 emotions depending on confidence.
    If the top emotion has confidence >= 0.70, returns a list with only the top emotion.
    Otherwise, returns a list containing the top two emotions.
    Compatible with router.py.
    """
    global _emotion_classifier
    if _emotion_classifier is None:
        _emotion_classifier = EmotionClassifier()
    res = _emotion_classifier.predict(text)
    
    emotion = res.get("emotion", "Sadness")
    confidence = res.get("confidence", 0.0)
    
    if confidence >= 0.70:
        return [emotion]
    else:
        # Find the second highest scoring emotion from the all_scores dictionary
        all_scores = res.get("all_scores", {})
        sorted_scores = sorted(
            [(k, v) for k, v in all_scores.items() if k != emotion],
            key=lambda x: x[1],
            reverse=True
        )
        if sorted_scores:
            second_emotion = sorted_scores[0][0]
            return [emotion, second_emotion]
        return [emotion]


def classify_intent(text: str, language: str = "English") -> str:
    """
    Classifies intent and returns the category name as a string (e.g. 'asking_mental_health_question').
    Compatible with router.py.
    """
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    res = _intent_classifier.classify(text, language)
    return res.type

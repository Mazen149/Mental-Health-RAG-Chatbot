"""
NLP Module exposing intent classification, language detection, and routing.
"""

from .intent_classifier import IntentClassifier, IntentRouter, classify_intent
from .language_detector import LanguageDetector, detect_language

__all__ = [
    "IntentClassifier",
    "IntentRouter",
    "classify_intent",
    "LanguageDetector",
    "detect_language",
]

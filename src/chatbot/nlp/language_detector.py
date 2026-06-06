"""
Language Detection Module
=========================
Detects the language of a given text using a pre-trained TF-IDF + Logistic Regression
model saved in `artifacts/langauge_detection/`.

Best model config (from notebook experiments):
    - TF-IDF: ngram_range=(1,5), analyzer='char', max_features=25000
    - Classifier: LogisticRegression(max_iter=1000)

Supported languages (20):
    pt, bg, zh, th, ru, en, vi, fr, nl, el,
    de, hi, it, ar, es, tr, sw, ur, pl, ja

Usage:
    from language_detector import LanguageDetector

    detector = LanguageDetector()
    result = detector.detect("Hello, how are you?")
    # result -> {"language": "en", "language_name": "English", "confidence": 0.97}
"""

import os
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_LEN = 300  # 95th percentile of training text lengths

# ISO 639-1 code → full language name (20 supported languages)
LANGUAGE_NAMES = {
    "ar": "Arabic",    "bg": "Bulgarian", "de": "German",     "el": "Greek",
    "en": "English",   "es": "Spanish",   "fr": "French",     "hi": "Hindi",
    "it": "Italian",   "ja": "Japanese",  "nl": "Dutch",      "pl": "Polish",
    "pt": "Portuguese","ru": "Russian",   "sw": "Swahili",    "th": "Thai",
    "tr": "Turkish",   "ur": "Urdu",      "vi": "Vietnamese", "zh": "Chinese",
}

# Dynamically locate the project root by searching upwards for .env or pyproject.toml
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parents[2]  # Fallback

_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "language_detection_best_model.pkl"
_DEFAULT_VECTORIZER_PATH = _PROJECT_ROOT / "artifacts" / "language_detection_best_vectorizer.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (must match the notebook's preprocessing exactly)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Clean and normalise text for language detection.

    Pipeline:
        1. Remove URLs and emails
        2. Remove numbers
        3. Trim to MAX_LEN characters
        4. Unicode NFKC normalisation
        5. Lowercase
        6. Remove punctuation (replace with spaces)
        7. Collapse whitespace
        8. Strip leading/trailing spaces
    """
    # 1. Remove URLs and Emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)

    # 2. Remove Numbers
    text = re.sub(r'\d+', '', text)

    # 3. Trim text to MAX_LEN characters
    text = text[:MAX_LEN]

    # 4. Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # 5. Lowercase
    text = text.lower()

    # 6. Remove punctuation (replace symbols with spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 7. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # 8. Strip
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Language Detector Class
# ─────────────────────────────────────────────────────────────────────────────

class LanguageDetector:
    """
    Detects the language of input text using a pre-trained TF-IDF + Logistic Regression model.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the pickled scikit-learn model file.
    vectorizer_path : str or Path, optional
        Path to the pickled TF-IDF vectorizer file.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        vectorizer_path: str | Path = _DEFAULT_VECTORIZER_PATH,
    ):
        model_path = Path(model_path)
        vectorizer_path = Path(vectorizer_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Run the language_detection_notebook to train and save the model first."
            )
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {vectorizer_path}\n"
                "Run the language_detection_notebook to train and save the vectorizer first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def detect(self, text: str) -> dict:
        """
        Detect the language of the given text.

        Parameters
        ----------
        text : str
            Raw input text (any language).

        Returns
        -------
        dict
            {
                "language": str,         # ISO 639-1 code (e.g. "en")
                "language_name": str,    # Full name (e.g. "English")
                "confidence": float,     # Prediction probability (0.0–1.0)
            }
        """
        cleaned = preprocess(text)
        if not cleaned:
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
            }

        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]

        # Get confidence from prediction probabilities
        probabilities = self.model.predict_proba(X)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])

        return {
            "language": prediction,
            "language_name": LANGUAGE_NAMES.get(prediction, "Unknown"),
            "confidence": round(confidence, 4),
        }

    def detect_top_k(self, text: str, k: int = 3) -> list[dict]:
        """
        Return top-k language predictions with confidence scores.

        Parameters
        ----------
        text : str
            Raw input text.
        k : int
            Number of top predictions to return.

        Returns
        -------
        list[dict]
            Sorted by confidence (descending).
        """
        cleaned = preprocess(text)
        if not cleaned:
            return [{"language": "unknown", "language_name": "Unknown", "confidence": 0.0}]

        X = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(X)[0]

        top_indices = np.argsort(probabilities)[::-1][:k]
        results = []
        for idx in top_indices:
            lang_code = self.model.classes_[idx]
            results.append({
                "language": lang_code,
                "language_name": LANGUAGE_NAMES.get(lang_code, "Unknown"),
                "confidence": round(float(probabilities[idx]), 4),
            })

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (module-level)
# ─────────────────────────────────────────────────────────────────────────────

_detector_instance = None


def detect_language(text: str) -> dict:
    """
    Convenience function: detect language without manually creating a LanguageDetector.

    Uses a lazily-initialised singleton instance.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LanguageDetector()
    return _detector_instance.detect(text)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    detector = LanguageDetector()

    test_cases = [
        "Hello, how are you?",
        "Bonjour, comment allez-vous?",
        "مرحبا، كيف حالك؟",
        "你好，你好吗？",
        "こんにちは",
        "Привет, как дела?",
        "Hola, ¿cómo estás?",
        "สวัสดีครับ",
    ]

    print("=" * 70)
    print(" LANGUAGE DETECTION — Quick Test")
    print("=" * 70)
    for text in test_cases:
        result = detector.detect(text)
        print(
            f"  [{result['language']}] {result['language_name']:<12} "
            f"(conf={result['confidence']:.4f})  "
            f"<- \"{text}\""
        )
    print("=" * 70)

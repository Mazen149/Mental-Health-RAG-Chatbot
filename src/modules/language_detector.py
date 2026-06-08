import os
import re
import unicodedata
from pathlib import Path
import joblib
import numpy as np

# ISO 639-1 code -> full language name (20 supported languages)
LANGUAGE_NAMES = {
    "ar": "Arabic",    "bg": "Bulgarian", "de": "German",     "el": "Greek",
    "en": "English",   "es": "Spanish",   "fr": "French",     "hi": "Hindi",
    "it": "Italian",   "ja": "Japanese",  "nl": "Dutch",      "pl": "Polish",
    "pt": "Portuguese","ru": "Russian",   "sw": "Swahili",    "th": "Thai",
    "tr": "Turkish",   "ur": "Urdu",      "vi": "Vietnamese", "zh": "Chinese",
}

# Locate project root
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parents[2]

_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "langauge_detection" / "language_detection_best_model.pkl"
_DEFAULT_VECTORIZER_PATH = _PROJECT_ROOT / "artifacts" / "langauge_detection" / "language_detection_best_vectorizer.pkl"


MAX_LEN = 300


def preprocess(text: str) -> str:
    """Comprehensive preprocessing from notebooks/language_detection.ipynb."""
    # 1. Remove URLs and Emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)

    # 2. Remove Numbers (Language-agnostic noise)
    text = re.sub(r'\d+', '', text)

    # 3. Trim text to a maximum of 300 characters
    text = text[:MAX_LEN]

    # 4. Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # 5. Lower casing
    text = text.lower()

    # 6. Remove punctuation (Replaces symbols with spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 7. Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # 8. Strip leading/trailing spaces
    return text.strip()



class LanguageDetector:
    """Language detector using model and vectorizer from Language_Detection.ipynb."""

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        vectorizer_path: str | Path = _DEFAULT_VECTORIZER_PATH,
    ):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def detect(self, text: str) -> dict:
        cleaned = preprocess(text)
        if not cleaned:
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
            }

        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])

        return {
            "language": prediction,
            "language_name": LANGUAGE_NAMES.get(prediction, "Unknown"),
            "confidence": round(confidence, 4),
        }

    def detect_top_k(self, text: str, k: int = 3) -> list[dict]:
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


# Singleton convenience instance
_detector_instance = None


def detect_language(text: str) -> dict:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LanguageDetector()
    return _detector_instance.detect(text)

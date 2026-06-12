import os
import re
import unicodedata
from pathlib import Path
import joblib
import numpy as np

# Apply monkeypatch to fix fasttext-wheel / numpy 2.x compatibility issue.
# FastText uses np.array(probs, copy=False) which fails on numpy 2.x.
_orig_array = np.array
def _patched_array(object, *args, **kwargs):
    if kwargs.get("copy") is False:
        kwargs.pop("copy")
    return _orig_array(object, *args, **kwargs)
np.array = _patched_array

import fasttext
from ..config import config

# ISO 639-1 code -> full language name (20 supported languages)
LANGUAGE_NAMES = {
    "ar": "Arabic",    "bg": "Bulgarian", "de": "German",     "el": "Greek",
    "en": "English",   "es": "Spanish",   "fr": "French",     "hi": "Hindi",
    "it": "Italian",   "ja": "Japanese",  "nl": "Dutch",      "pl": "Polish",
    "pt": "Portuguese","ru": "Russian",   "sw": "Swahili",    "th": "Thai",
    "tr": "Turkish",   "ur": "Urdu",      "vi": "Vietnamese", "zh": "Chinese",
}

MAX_LEN = config.LANGUAGE_DETECTION_MAX_LEN


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
    """Language detector using custom model and vectorizer with FastText fallback."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        vectorizer_path: str | Path | None = None,
        fasttext_model_path: str | Path | None = None,
    ):
        self.model_path = Path(model_path) if model_path else config.MOD1_CLASSIFIER_PATH
        self.vectorizer_path = Path(vectorizer_path) if vectorizer_path else config.MOD1_VECTORIZER_PATH
        self.fasttext_model_path = Path(fasttext_model_path) if fasttext_model_path else config.ARTIFACTS_DIR / "lid.176.ftz"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {self.vectorizer_path}")
        if not self.fasttext_model_path.exists():
            raise FileNotFoundError(f"FastText model file not found: {self.fasttext_model_path}")

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.ft_model = fasttext.load_model(str(self.fasttext_model_path))

    def detect(self, text: str, threshold: float | None = None) -> dict:
        cleaned = preprocess(text)
        if not cleaned:
            return {
                "language": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0,
            }

        # 1. Predict using custom TF-IDF model
        X = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(X)[0]

        probabilities = self.model.predict_proba(X)[0]
        class_idx = list(self.model.classes_).index(prediction)
        confidence = float(probabilities[class_idx])

        # Strip router padding tokens if present to check true word count
        unpadded = text.replace("/p", "").replace("/P", "").strip()

        # Default confidence threshold is 0.50
        if threshold is None:
            threshold = 0.50

        # 2. Fallback to FastText if confidence is below threshold
        if confidence < threshold:
            try:
                if unpadded:
                    lbls, probs = self.ft_model.predict(unpadded, k=1)
                    if lbls and probs:
                        pred_lang = lbls[0].replace("__label__", "")
                        if pred_lang in LANGUAGE_NAMES:
                            prediction = pred_lang
                            confidence = float(probs[0])
            except Exception:
                pass

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


def detect_language(text: str, threshold: float | None = None) -> dict:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LanguageDetector()
    return _detector_instance.detect(text, threshold=threshold)

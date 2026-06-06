"""
Emotion Classifier Module
=========================
Detects the emotional state of a user's message using a pre-trained XLM-RoBERTa-base + LoRA
model saved in `artifacts/emotion_classification/`.

Classes:
    0: "Sadness", 1: "Joy", 2: "Love",
    3: "Anger",   4: "Fear", 5: "Surprise"

Usage:
    from chatbot.nlp.emotion_classifier import EmotionClassifier

    classifier = EmotionClassifier()
    result = classifier.predict("I am feeling very anxious today.")
    # result -> {"emotion": "Fear", "confidence": 0.95}
"""

import os
from pathlib import Path
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Dynamically locate the project root by searching upwards for .env or pyproject.toml
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parents[2]  # Fallback

_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "emotion_classifier"

# Load environment variables (like HF_TOKEN) from .env file
_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

# Emotion mapping
EMOTION_MAP = {
    0: "Sadness", 1: "Joy", 2: "Love",
    3: "Anger",   4: "Fear", 5: "Surprise"
}

# ─────────────────────────────────────────────────────────────────────────────
# Emotion Classifier Class
# ─────────────────────────────────────────────────────────────────────────────

class EmotionClassifier:
    """
    Detects the emotion of input text using a fine-tuned XLM-RoBERTa + LoRA model.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the directory containing the LoRA adapter and tokenizer.
    """

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_path}\n"
                "Ensure the model is downloaded or trained and placed in the artifacts folder."
            )
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model (uses HF_TOKEN if available)
        hf_token = os.getenv("HF_TOKEN")
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "xlm-roberta-base",
            num_labels=len(EMOTION_MAP),
            dtype=torch.float32,
            ignore_mismatched_sizes=True,
            token=hf_token
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def predict(self, text: str) -> dict:
        """
        Detect the emotion of the given text.

        Parameters
        ----------
        text : str
            Raw input text (any language).

        Returns
        -------
        dict
            {
                "emotion": str,          # Emotion label (e.g. "Sadness")
                "emotion_id": int,       # Internal label ID (e.g. 0)
                "confidence": float,     # Prediction probability (0.0-1.0)
                "all_scores": dict       # Probabilities for all classes
            }
        """
        if not text or not text.strip():
            return {
                "emotion": "Unknown",
                "emotion_id": -1,
                "confidence": 0.0,
                "all_scores": {}
            }

        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_emotion = EMOTION_MAP[pred_idx]
        confidence = float(probs[pred_idx])
        
        all_scores = {EMOTION_MAP[i]: round(float(probs[i]), 4) for i in range(len(EMOTION_MAP))}

        return {
            "emotion": pred_emotion,
            "emotion_id": pred_idx,
            "confidence": round(confidence, 4),
            "all_scores": all_scores
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (module-level)
# ─────────────────────────────────────────────────────────────────────────────

_classifier_instance = None


def detect_emotion(text: str) -> dict:
    """
    Convenience function: detect emotion without manually creating an EmotionClassifier.
    Uses a lazily-initialised singleton instance.
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmotionClassifier()
    return _classifier_instance.predict(text)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    try:
        classifier = EmotionClassifier()

        test_cases = [
            "I feel so empty and hopeless, nothing matters anymore.",
            "I am furious, this is completely unacceptable!",
            "My heart is pounding, I think something terrible happened.",
            "I am absolutely speechless, I never thought I'd win!",
            "Tu me combles de bonheur, je t'aime tellement.",
        ]

        print("=" * 70)
        print(" EMOTION CLASSIFICATION — Quick Test")
        print("=" * 70)
        for text in test_cases:
            result = classifier.predict(text)
            print(
                f"  [{result['emotion']}] "
                f"(conf={result['confidence']:.4f})  "
                f"<- \"{text}\""
            )
        print("=" * 70)
    except FileNotFoundError as e:
        print(f"Skipping quick test: {e}")

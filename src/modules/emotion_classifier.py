import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from ..config import config

EMOTION_MAP = {
    0: "Sadness", 1: "Joy", 2: "Love",
    3: "Anger",   4: "Fear", 5: "Surprise"
}

class EmotionClassifier:
    """Emotion classifier using XLM-RoBERTa-base ONNX exported model."""

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else config.MOD2_DIR
        model_path = self.model_dir / "model.onnx"
        
        if not model_path.exists():
            print(f"ONNX model not found at {model_path}. Please export it first.")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"--> [Emotion Classifier] Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(str(model_path))
        
        tokenizer_path = config.MOD2_DIR / "tokenizer.json"
        if not tokenizer_path.exists():
            tokenizer_path = self.model_dir / "tokenizer.json"
        
        print(f"--> [Emotion Classifier] Loading tokenizer from {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        self.tokenizer.enable_truncation(max_length=64)
        self.tokenizer.enable_padding(length=64)

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "emotion": "Unknown",
                "emotion_id": -1,
                "confidence": 0.0,
                "all_scores": {}
            }

        encoded = self.tokenizer.encode(text)
        
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        logits = self.session.run(None, ort_inputs)[0]
        # softmax
        exp_logits = np.exp(logits[0] - np.max(logits[0]))
        probs = exp_logits / exp_logits.sum()

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


# Singleton convenience instance
_classifier_instance = None


def detect_emotion(text: str) -> dict:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmotionClassifier()
    return _classifier_instance.predict(text)

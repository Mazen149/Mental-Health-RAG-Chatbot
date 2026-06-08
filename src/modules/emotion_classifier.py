import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from dotenv import load_dotenv

# Locate project root and load environment
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parents[2]

_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "emotion_classifier"

EMOTION_MAP = {
    0: "Sadness", 1: "Joy", 2: "Love",
    3: "Anger",   4: "Fear", 5: "Surprise"
}


class EmotionClassifier:
    """Emotion classifier using XLM-RoBERTa-base + LoRA from emotion-classifier.ipynb."""

    def __init__(self, model_path: str | Path = _DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.getenv("HF_TOKEN")

        base_model = AutoModelForSequenceClassification.from_pretrained(
            os.getenv("EMOTION_BASE_MODEL", "xlm-roberta-base"),
            num_labels=len(EMOTION_MAP),
            dtype=torch.float32,
            ignore_mismatched_sizes=True,
            token=hf_token
        )

        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def predict(self, text: str) -> dict:
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


# Singleton convenience instance
_classifier_instance = None


def detect_emotion(text: str) -> dict:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = EmotionClassifier()
    return _classifier_instance.predict(text)

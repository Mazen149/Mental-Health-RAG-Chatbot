import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.modules.emotion_classifier import EmotionClassifier, detect_emotion


@pytest.fixture
def mock_classifier():
    """Fixture that mocks the heavy model loading and returns a dummy classifier."""
    with (
        patch("src.modules.emotion_classifier.ort.InferenceSession") as mock_session,
        patch("src.modules.emotion_classifier.Tokenizer.from_file") as mock_tokenizer,
        patch("src.modules.emotion_classifier.Path.exists", return_value=True),
    ):
        # Setup mock tokenizer
        mock_tok_instance = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [1, 2, 3]
        mock_encoding.attention_mask = [1, 1, 1]
        mock_tok_instance.encode.return_value = mock_encoding
        mock_tokenizer.return_value = mock_tok_instance

        # Setup mock session
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        yield EmotionClassifier()


def test_missing_model_directory():
    """Verify that a FileNotFoundError is raised if the model path doesn't exist."""
    with patch("src.modules.emotion_classifier.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            EmotionClassifier(model_dir="fake/path")


def test_predict_empty_text(mock_classifier):
    """Test behavior when empty text is provided."""
    result = mock_classifier.predict("")
    assert result["emotion"] == "Unknown"
    assert result["confidence"] == 0.0


def test_predict_standard(mock_classifier):
    """Test standard emotion prediction with mocked logits."""
    # Mock the model output
    # Let's say label 0 (Sadness) has the highest logit
    dummy_logits = np.array([[[10.0, 1.0, -1.0, 0.0, -2.0, -3.0]]])
    mock_classifier.session.run.return_value = dummy_logits

    result = mock_classifier.predict("I feel so sad and lonely.")

    assert result["emotion"] == "Sadness"
    assert result["emotion_id"] == 0
    assert result["confidence"] > 0.99
    assert result["all_scores"]["Sadness"] > 0.99


def test_detect_emotion_convenience():
    """Test the module-level singleton convenience function."""
    with patch("src.modules.emotion_classifier.EmotionClassifier") as mock_init:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = {
            "emotion": "Anger",
            "emotion_id": 3,
            "confidence": 0.90,
            "all_scores": {},
        }
        mock_init.return_value = mock_instance

        result = detect_emotion("I am furious!")

        assert result["emotion"] == "Anger"
        assert result["confidence"] == 0.90
        mock_init.assert_called_once()

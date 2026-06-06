import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

from src.modules.emotion_classifier import EmotionClassifier, detect_emotion, EMOTION_MAP

@pytest.fixture
def mock_classifier():
    """Fixture that mocks the heavy model loading and returns a dummy classifier."""
    with patch('src.modules.emotion_classifier.AutoModelForSequenceClassification.from_pretrained') as mock_base, \
         patch('src.modules.emotion_classifier.PeftModel.from_pretrained') as mock_peft, \
         patch('src.modules.emotion_classifier.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('src.modules.emotion_classifier.Path.exists', return_value=True):
        
        # Setup mock tokenizer
        mock_tok_instance = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.to.return_value = mock_encoding
        mock_tok_instance.return_value = mock_encoding
        mock_tokenizer.return_value = mock_tok_instance
        
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_peft.return_value = mock_model_instance
        
        yield EmotionClassifier()

def test_missing_model_directory():
    """Verify that a FileNotFoundError is raised if the model path doesn't exist."""
    with patch('src.modules.emotion_classifier.Path.exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            EmotionClassifier(model_path="fake/path")

def test_predict_empty_text(mock_classifier):
    """Test behavior when empty text is provided."""
    result = mock_classifier.predict("")
    assert result["emotion"] == "Unknown"
    assert result["confidence"] == 0.0

@patch('src.modules.emotion_classifier.torch.softmax')
@patch('src.modules.emotion_classifier.torch.no_grad')
def test_predict_standard(mock_no_grad, mock_softmax, mock_classifier):
    """Test standard emotion prediction with mocked logits/softmax."""
    # Mock the model output
    mock_logits = MagicMock()
    mock_classifier.model.return_value = MagicMock(logits=mock_logits)
    
    # Mock softmax to return a dummy probability distribution
    # Let's say label 0 (Sadness) has the highest probability (0.85)
    dummy_probs = MagicMock()
    dummy_probs_cpu = MagicMock()
    dummy_probs_cpu.numpy.return_value = np.array([0.85, 0.05, 0.02, 0.05, 0.01, 0.02])
    dummy_probs.__getitem__.return_value.cpu.return_value = dummy_probs_cpu
    mock_softmax.return_value = dummy_probs
    
    result = mock_classifier.predict("I feel so sad and lonely.")
    
    assert result["emotion"] == "Sadness"
    assert result["emotion_id"] == 0
    assert result["confidence"] == 0.85
    assert result["all_scores"]["Sadness"] == 0.85
    assert result["all_scores"]["Joy"] == 0.05

@patch('src.modules.emotion_classifier.torch.softmax')
@patch('src.modules.emotion_classifier.torch.no_grad')
def test_detect_emotion_convenience(mock_no_grad, mock_softmax):
    """Test the module-level singleton convenience function."""
    with patch('src.modules.emotion_classifier.EmotionClassifier') as mock_init:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = {
            "emotion": "Anger",
            "emotion_id": 3,
            "confidence": 0.90,
            "all_scores": {}
        }
        mock_init.return_value = mock_instance
        
        result = detect_emotion("I am furious!")
        
        assert result["emotion"] == "Anger"
        assert result["confidence"] == 0.90
        mock_init.assert_called_once()

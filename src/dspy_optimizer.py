import json
import os
import sys
from pathlib import Path

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from dspy.teleprompt import BootstrapFewShot
from src.config import config
from src.modules.prompts import IntentClassifierModule

def get_train_examples():
    """Create a dataset of few-shot examples for intent classification."""
    raw_examples = [
        # Greetings
        {"text": "hello, how are you", "type": "greeting"},
        {"text": "مرحبا، كيف حالك؟", "type": "greeting"}, # Arabic
        {"text": "hola, qué tal?", "type": "greeting"}, # Spanish
        {"text": "ہیلو، آپ کیسے ہیں؟", "type": "greeting"}, # Urdu
        {"text": "bonjour, comment ça va?", "type": "greeting"}, # French
        
        # Goodbye
        {"text": "see you later", "type": "goodbye"},
        {"text": "مع السلامة", "type": "goodbye"}, # Arabic
        {"text": "adiós, cuídate", "type": "goodbye"}, # Spanish
        {"text": "خدا حافظ", "type": "goodbye"}, # Urdu
        {"text": "au revoir", "type": "goodbye"}, # French
        
        # Gratitude
        {"text": "thank you so much", "type": "gratitude"},
        {"text": "شكرا جزيلا لك", "type": "gratitude"}, # Arabic
        {"text": "muchas gracias por tu ayuda", "type": "gratitude"}, # Spanish
        {"text": "بہت شکریہ", "type": "gratitude"}, # Urdu
        {"text": "thx sm", "type": "gratitude"}, # Slang
        
        # Out of Scope
        {"text": "what's the weather like in New York?", "type": "out_of_scope"},
        {"text": "كم سعر البيتكوين اليوم؟", "type": "out_of_scope"}, # Arabic
        {"text": "cómo se hace una paella?", "type": "out_of_scope"}, # Spanish
        {"text": "write a python script to sort an array", "type": "out_of_scope"},
        {"text": "meu computador quebrou, como consertar?", "type": "out_of_scope"}, # Portuguese
        {"text": "My boss yelled at me so I need to hire a lawyer.", "type": "out_of_scope"}, # Edge case
        
        # Asking Mental Health Question
        {"text": "I've been feeling really anxious lately and I can't sleep.", "type": "asking_mental_health_question"},
        {"text": "كيف أتعامل مع نوبات الهلع؟", "type": "asking_mental_health_question"}, # Arabic
        {"text": "me siento muy deprimido y sin motivación.", "type": "asking_mental_health_question"}, # Spanish
        {"text": "مجھے ہر وقت ڈر لگتا ہے اور نیند نہیں آتی", "type": "asking_mental_health_question"}, # Urdu
        {"text": "im so done with everything right now", "type": "asking_mental_health_question"}, # Edge case
        {"text": "je suis tellement stressé par mes examens", "type": "asking_mental_health_question"}, # French
        
        # Crisis
        {"text": "I don't want to live anymore.", "type": "crisis"},
        {"text": "أريد أن أؤذي نفسي", "type": "crisis"}, # Arabic
        {"text": "quiero suicidarme", "type": "crisis"}, # Spanish
        {"text": "میں خودکشی کرنا چاہتا ہوں", "type": "crisis"}, # Urdu
        {"text": "je veux me tuer", "type": "crisis"}, # French
        {"text": "i wanna kms", "type": "crisis"}, # Slang edge case
    ]
    
    examples = []
    for ex in raw_examples:
        # We specify text as the input field. The target field is type.
        examples.append(dspy.Example(text=ex["text"], type=ex["type"]).with_inputs("text"))
        
    return examples

def intent_exact_match(example, pred, trace=None):
    """Metric function: Exact match on the intent type."""
    pred_type = pred["type"] if isinstance(pred, dict) else pred.type
    return example.type.lower() == str(pred_type).lower()

def optimize_intent_classifier():
    print("--> Setting up DSPy optimization for IntentClassifierModule...")
    
    # Ensure artifacts directory exists
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Initialize LM
    groq_api_key = config.GROQ_API_KEY
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY is not set in environment or config.")
        return
        
    lm = dspy.LM(f"groq/{config.GROQ_CLASSIFIER_MODEL}", api_key=groq_api_key)
    dspy.settings.configure(lm=lm)
    
    # Get training examples
    trainset = get_train_examples()
    
    # Initialize the unoptimized module
    module = IntentClassifierModule()
    
    # Set up the BootstrapFewShot optimizer
    # This optimizer generates examples (traces) of the reasoning process
    # and keeps the successful ones as few-shot prompts.
    optimizer = BootstrapFewShot(
        metric=intent_exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=8
    )
    
    print(f"--> Starting compilation with {len(trainset)} examples...")
    compiled_module = optimizer.compile(module, trainset=trainset)
    
    # Save the optimized prompt parameters
    output_path = artifacts_dir / "intent_classifier_optimized.json"
    compiled_module.save(str(output_path))
    print(f"--> Optimization complete! Saved optimized module to {output_path}")

if __name__ == "__main__":
    optimize_intent_classifier()

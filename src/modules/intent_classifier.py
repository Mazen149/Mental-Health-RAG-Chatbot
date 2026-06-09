import json
import os
from pathlib import Path
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import dspy
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import numpy as np
from pydantic import BaseModel, Field

# Locate project root and load environment
from ..config import config


class Intent(BaseModel):
    type: Literal["greeting", "goodbye", "gratitude", "out_of_scope", "asking_mental_health_question", "crisis"] = Field(
        ...,
        description="greeting: for greeting, introduction, small talk; goodbye: for farewell; gratitude: for expressions of thanks; out_of_scope: for off-topic queries; asking_mental_health_question: for clinical mental health queries; crisis: for queries indicating self-harm, suicide, or crisis",
    )
    confidence: float = Field(
        ...,
        description="confidence score of the intent classification, between 0 and 1",
    )
    classifier: Literal["embedding", "llm"] = Field(
        ...,
        description="the classifier used for intent classification: embedding: a classifier based on sentence embeddings and cosine similarity; llm: a classifier based on a large language model",
    )


from .prompts import IntentClassifierModule


class IntentClassifier:
    """Intent classifier using embeddings and LLM fallback."""

    def __init__(self, llm_fallback: Callable[[str], Intent] | None = None):
        self.llm_fallback = llm_fallback or self._llm_fallback

        self.model_name = config.GROQ_CLASSIFIER_MODEL

        # Initialize DSPy LM if API key is available
        groq_api_key = config.GROQ_API_KEY
        if groq_api_key:
            self.lm = dspy.LM(f"groq/{self.model_name}", api_key=groq_api_key)
            self.fallback_module = IntentClassifierModule()
            self.groq_client = self.lm  # keep for backwards compatibility / mock checks
        else:
            self.lm = None
            self.fallback_module = None
            self.groq_client = None


        self.embedding_examples = {
            "greeting": [
                "hello", "hi there", "hey", "greetings", "good morning", "good afternoon", "good evening", "hiya",
                "hello, how are you", "how are you", "how's it going", "how are you doing", "how are yoy", "nice to meet you",
                "my name is john", "i am john", "call me john", "what is my name", "who am i", "my name is", "tell me my name",
                "who are you", "what is your name", "what can you do", "how can you help me",
                "مرحبا", "أهلاً", "اهلا", "السلام عليكم", "سلام عليكم", "صباح الخير", "كيف حالك", "اسمي احمد", "انا احمد", "ما هو اسمي", "من انت", "ما اسمك",
                "ہیلو", "السلام علیکم", "آپ کیسے ہیں", "میرا نام علی ہے", "میرا نام کیا ہے",
                "hola", "buenos días", "buenas tardes", "buenas noches", "bonjour", "salut", "bonsoir", "ciao", "buongiorno", "你好"
            ],
            "goodbye": [
                "goodbye", "bye", "see you", "take care", "farewell", "bye bye",
                "مع السلامة", "إلى اللقاء", "وداعا", "خدا حافظ",
                "tschüss", "auf wiedersehen", "au revoir", "adiós", "arrivederci", "再见"
            ],
            "gratitude": [
                "thanks", "thank you", "thx", "appreciate it", "thank you so much",
                "شكرا", "شكراً", "تسلم", "جزاك الله خيراً", "شکریہ", "بہت شکریہ",
                "danke", "merci", "gracias", "grazie", "asante", "谢谢"
            ],
            "out_of_scope": [
                "what is the weather like", "tell me a joke", "what is the latest news", "what sports scores are",
                "ما هو الطقس اليوم", "اخبرني بنكتة", "ما هي آخر الأخبار",
                "آج موسم کیسا ہے", "مجھے لطیفہ سنائیں", "تازہ ترین خبریں کیا ہیں",
                "quel temps fait-il", "raconte-moi une blague", "cómo está el clima", "cuéntame un chiste"
            ],
            "asking_mental_health_question": [
                "I feel anxious and need help", "I am depressed", "can you help me with my stress", "I need therapy advice",
                "أشعر بالقلق وأحتاج إلى مساعدة", "أنا مكتئب وأحتاج للتحدث مع شخص ما", "ساعدني في التغلب على التوتر",
                "میں فکر مند محسوس کرتا ہوں اور مجھے مدد کی ضرورت ہے", "میں اداس اور افسردہ ہوں", "کیا آپ میرے تناؤ میں مدد کر سکتے ہیں",
                "je me sens anxieux", "je suis déprimé", "me siento ansioso", "estoy deprimido"
            ],
            "crisis": [
                "I want to kill myself", "I want to end my life", "I am thinking of suicide", "I want to cut myself", "how to end my life", "harm myself",
                "أريد الانتحار", "أريد إنهاء حياتي", "أفكر في الانتحار", "أريد إيذاء نفسي",
                "میں خودکشی کرنا چاہتا ہوں", "میں اپنی زندگی ختم کرنا چاہتا ہوں", "میں خود کو نقصان پہنچانا چاہتا ہوں",
                "je veux me tuer", "je veux en glorie avec ma vie", "quiero suicidarme", "quiero terminar con mi vida"
            ],
        }

        self.embedding_threshold = 0.65
        self.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        hf_token = config.HF_TOKEN
        self.embedding_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )

        try:
            from sentence_transformers import SentenceTransformer
            print("--> [Intent Classifier] Loading local sentence-transformer model...")
            self.local_model = SentenceTransformer(self.embedding_model)
            self.use_local = True
        except Exception as e:
            print(f"--> [Intent Classifier] Local sentence-transformer not used: {e}. Using HF API Client.")
            self.use_local = False

        # Precompute normalized embeddings for example sentences
        if self.use_local:
            self.embedding_examples_embeddings = {
                intent_type: self._normalize_embeddings(
                    np.asarray(
                        self.local_model.encode(examples),
                        dtype=np.float32,
                    )
                )
                for intent_type, examples in self.embedding_examples.items()
            }
        else:
            self.embedding_examples_embeddings = {
                intent_type: self._normalize_embeddings(
                    np.asarray(
                        self.embedding_client.feature_extraction(
                            examples,
                            model=self.embedding_model,
                        ),
                        dtype=np.float32,
                    )
                )
                for intent_type, examples in self.embedding_examples.items()
            }

    def classify(self, text: str, language: str = "English") -> Intent:
        # Check crisis/self-harm keywords first as an instant safety bypass for all languages!
        from .rag import detect_crisis
        if detect_crisis(text):
            return Intent(type="crisis", confidence=0.95, classifier="embedding")

        # Run embedding classifier for all languages since we now use a multilingual embedding model
        embedding_intent = self._embedding_classifier(text)
        if embedding_intent is not None:
            print(f"--> [Intent Classifier] Classified intent as '{embedding_intent.type}' with confidence {embedding_intent.confidence:.5f} using embedding classifier.")
            return embedding_intent

        # LLM fallback
        return self.llm_fallback(text)

    def _embedding_classifier(self, text: str) -> Intent | None:
        text_embedding = self._get_embedding(text)
        best_intent = None
        best_score = 0.0

        for intent_type, example_embeddings in self.embedding_examples_embeddings.items():
            if example_embeddings.size == 0:
                continue
            scores = example_embeddings @ text_embedding
            max_score = float(np.max(scores))
            if max_score > best_score:
                best_score = max_score
                best_intent = intent_type

        if best_intent is not None and best_score >= self.embedding_threshold:
            return Intent(
                type=best_intent,
                confidence=best_score,
                classifier="embedding",
            )
        return None

    def _llm_fallback(self, text: str) -> Intent:
        # Try DSPy predictor as the robust fallback
        if self.fallback_module is not None and self.lm is not None:
            try:
                with dspy.context(lm=self.lm):
                    res = self.fallback_module(text=text)
                print(f"--> [Intent Classifier] Received DSPy response: {res}")
                return Intent(
                    type=res["type"],
                    confidence=res["confidence"],
                    classifier="llm"
                )
            except Exception as e:
                try:
                    print(f"--> [Intent Classifier Error] DSPy API fallback failed: {e}")
                except Exception:
                    pass

        # Return default if Groq fails or is not available
        return Intent(type="greeting", confidence=0.5, classifier="llm")

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.use_local:
            embedding = self.local_model.encode(text)
        else:
            embedding = self.embedding_client.feature_extraction(
                text,
                model=self.embedding_model,
            )
        return self._normalize_embeddings(np.asarray(embedding, dtype=np.float32))

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            norm = np.linalg.norm(embeddings)
            return embeddings / norm if norm > 0 else embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


# Singleton convenience instance
_classifier_instance = None


def classify_intent(message: str, language: str = "English") -> Intent:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance.classify(message, language)


# Redirection router wrapper for backend compatibility
class IntentRouter:
    """Simple wrapper to router general, out_of_scope, and asking_mental_health_question."""

    def __init__(self):
        self.classifier = IntentClassifier()

    def route(self, user_message: str, detected_language: str = "en") -> dict:
        intent_res = self.classifier.classify(user_message, detected_language)
        
        # Simple mapping to original routing dictionary
        intent = intent_res.type
        use_rag = (intent == "asking_mental_health_question")
        
        action = "direct_reply"
        if intent == "asking_mental_health_question":
            action = "rag_pipeline"
        elif intent == "out_of_scope":
            action = "decline"

        # Map to template response as a fallback
        template = None
        if intent == "greeting":
            template = "Hello! I'm here to support you with mental health topics. Feel free to share what's on your mind. 😊"
        elif intent == "goodbye":
            template = "Goodbye! Take care of yourself. I am always here if you need support."
        elif intent == "gratitude":
            template = "You're welcome! I'm glad I could help."
        elif intent == "out_of_scope":
            template = "I'm specialised in mental health support, so I can't help with that topic. If you have questions about anxiety, depression, stress, or emotional wellbeing, I'm here for you! 💙"

        return {
            "user_message": user_message,
            "detected_language": detected_language,
            "intent": intent,
            "confidence": intent_res.confidence,
            "stage": intent_res.classifier,
            "action": action,
            "use_rag": use_rag,
            "response": template,
        }

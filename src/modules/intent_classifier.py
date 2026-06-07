import json
import os
from pathlib import Path
from typing import Callable, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import numpy as np
from pydantic import BaseModel, Field

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


class Intent(BaseModel):
    type: Literal["general", "out_of_scope", "asking_mental_health_question", "crisis"] = Field(
        ...,
        description="general: for greeting, goodbye, thanks; out_of_scope: for off-topic queries; asking_mental_health_question: for clinical mental health queries; crisis: for queries indicating self-harm, suicide, or crisis",
    )
    confidence: float = Field(
        ...,
        description="confidence score of the intent classification, between 0 and 1",
    )
    classifier: Literal["embedding", "llm"] = Field(
        ...,
        description="the classifier used for intent classification: embedding: a classifier based on sentence embeddings and cosine similarity; llm: a classifier based on a large language model",
    )


class IntentClassifier:
    """Intent classifier using embeddings and LLM fallback."""

    def __init__(self, llm_fallback: Callable[[str], Intent] | None = None):
        self.llm_fallback = llm_fallback or self._llm_fallback

        # Initialize Groq client if available
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            from groq import Groq
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None

        self.embedding_examples = {
            "general": [
                "hello", "thanks", "goodbye", "hi there", "thank you",
                "مرحبا", "شكرا", "مع السلامة", "ہیلو", "شکریہ",
                "hola", "gracias", "adiós", "bonjour", "merci", "au revoir"
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
        
        hf_token = os.environ.get("HF_TOKEN")
        self.embedding_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )

        # Precompute normalized embeddings for example sentences
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
        # Try Groq API as the robust multilingual fallback
        if self.groq_client is not None:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": text}
                    ],
                    model="openai/gpt-oss-20b",
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                content = chat_completion.choices[0].message.content
                print(f"--> [Intent Classifier] Received LLM response: {content}")
                return self._parse_llm_intent(content)
            except Exception as e:
                try:
                    print(f"--> [Intent Classifier Error] Groq API fallback failed: {e}")
                except Exception:
                    pass

        # Return default if Groq fails or is not available
        return Intent(type="general", confidence=0.5, classifier="llm")

    def _parse_llm_intent(self, content: str) -> Intent:
        cleaned = str(content).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        data = json.loads(cleaned)
        intent_type = data.get("type", "general")
        if intent_type not in {"general", "out_of_scope", "asking_mental_health_question", "crisis"}:
            intent_type = "general"

        confidence = data.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.65
        confidence = max(0.0, min(1.0, confidence))

        return Intent(
            type=intent_type,
            confidence=confidence,
            classifier="llm",
        )

    def _system_prompt(self) -> str:
        return (
            "You are a strict intent classification engine for a mental-health support assistant. "
            "Classify the user's message into exactly one label: general, out_of_scope, asking_mental_health_question, or crisis. "
            "Use general only for greetings, goodbyes, and thanks. "
            "Use crisis for any query containing suicidal thoughts, self-harm, cutting, ending one's life, or intent to inflict harm on oneself. "
            "Use asking_mental_health_question for any other mental-health-related question, emotional distress, therapy, anxiety, depression, panic, stress, or loneliness. "
            "Use out_of_scope for everything else that is not about mental health. "
            "Return only valid JSON with exactly these keys: type, confidence, classifier. "
            "The classifier value must always be llm. "
            "Confidence must be a number from 0 to 1 reflecting how sure you are. "
            "Do not include markdown, code fences, commentary, or extra keys."
        )

    def _get_embedding(self, text: str) -> np.ndarray:
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

        # Map 'general' to 'greeting' template response as a fallback
        template = None
        if intent == "general":
            template = "Hello! I'm here to support you with mental health topics. Feel free to share what's on your mind. 😊"
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

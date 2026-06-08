"""
================================================================================
SERENE AI — CONVERSATIONAL ROUTING ENGINE
================================================================================
Routes queries dynamically using conversational fast-paths (regex checking),
crisis safety bypass triggers, and fallbacks to full semantic RAG logic.
================================================================================
"""

import asyncio
import os
from pathlib import Path
import re

from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment Loading & Project Root Identification
# ------------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parent  # Fallback

_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

# Read toggle for query translation
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "False").lower() in ("true", "1", "yes")

# Local project imports
from .modules import detect_language, classify_emotion, classify_intent
from .modules.multilingual_patterns import (
    GREETING_REGEX, GOODBYE_REGEX, GRATITUDE_REGEX,
    GREETING_REGEX_BY_LANG, GOODBYE_REGEX_BY_LANG, GRATITUDE_REGEX_BY_LANG,
    GREETING_RESPONSES, GOODBYE_RESPONSES, GRATITUDE_RESPONSES,
    CRITICAL_CRISIS_RESPONSES, OUT_OF_SCOPE_RESPONSES
)
from .modules.rag import build_system_prompt, detect_crisis

# ------------------------------------------------------------------------------
# 2. Global Utilities & Helper Functions
# ------------------------------------------------------------------------------
def safe_print(msg: str) -> None:
    """Safe console logging print that handles UnicodeEncodeError on Windows command line."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

# Translation Model Pipeline Global
_translator_pipeline = None


def translate_to_english(text: str, client=None) -> str:
    """
    Translates non-English text to English using a local Helsinki-NLP translation transformer.
    Safely bypasses translation for pure ASCII inputs.
    """
    global _translator_pipeline
    try:
        # Fast path: if the text is pure ASCII, it is already English
        if all(ord(c) < 128 for c in text):
            return text

        if _translator_pipeline is None:
            from transformers import pipeline
            safe_print("--> [Translator Setup] Initializing local multilingual translation pipeline...")
            _translator_pipeline = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-mul-en",
                device=-1  # Force CPU execution
            )
        
        result = _translator_pipeline(text)
        translated = result[0]['translation_text'].strip()
        return translated if translated else text
    except Exception as e:
        safe_print(f"--> [Translator Error] Local translation failed: {e}")
        return text


def get_direct_greeting(language: str) -> str:
    """Returns a direct friendly greeting response in the user's language."""
    return GREETING_RESPONSES.get(language, GREETING_RESPONSES["English"])


def get_direct_goodbye(language: str) -> str:
    """Returns a direct friendly goodbye response in the user's language."""
    return GOODBYE_RESPONSES.get(language, GOODBYE_RESPONSES["English"])


def get_direct_gratitude(language: str) -> str:
    """Returns a direct warm gratitude acknowledgement in the user's language."""
    return GRATITUDE_RESPONSES.get(language, GRATITUDE_RESPONSES["English"])

async def route_query(query: str, rag_instance, history: list = None) -> dict:
    """
    Routes the user query dynamically in an asynchronous manner.
    
    Two-layer conversational intent detection:
      Layer 1: Regex fast-path for greeting/goodbye/gratitude (instant, 0ms)
      Layer 2: LLM classify_intent fallback for longer/complex messages
    
    INPUT: user query string
    OUTPUT: dict with answer, resources, and optional metadata fields
    """
    # =============================================
    # 0. LANGUAGE & CRISIS PRE-DETECTION (takes < 1ms locally)
    # =============================================
    query_words = query.strip().split()
    query_for_lang = query
    if len(query_words) < 5:
        query_for_lang = f"/p /p /p /p {query.strip()} /p /p /p /p" # Pad short queries to improve language detection accuracy with more context
    
    try:
        # Run local TF-IDF model synchronously (takes <1ms)
        language = detect_language(query_for_lang)
    except Exception as e:
        safe_print(f"Error in local language detection: {e}")
        language = "English"

    # =============================================
    # LAYER 1: Regex Fast-Path for Conversational Intents (20 languages)
    # =============================================
    cleaned = query.strip()
    
    if GREETING_REGEX.match(cleaned) and len(cleaned.split()) <= 2:
        safe_print("--> [Router] Matched greeting intent via Layer 1 regex.")
        # Exact language matching to bypass unreliable ML classification on short query text
        matched_language = None
        if language in GREETING_REGEX_BY_LANG and GREETING_REGEX_BY_LANG[language].match(cleaned):
            matched_language = language
        elif GREETING_REGEX_BY_LANG["English"].match(cleaned):
            matched_language = "English"
        else:
            for lang, r_pattern in GREETING_REGEX_BY_LANG.items():
                if r_pattern.match(cleaned):
                    matched_language = lang
                    break
        if matched_language:
            language = matched_language
            safe_print(f"--> [Router] Override language to {language} based on greeting pattern match.")
        return {
            "answer": get_direct_greeting(language),
            "resources": [],
            "language": None,
            "emotion": None,
            "intent": "greeting"
        }
    
    if GOODBYE_REGEX.match(cleaned) and len(cleaned.split()) <= 3:
        safe_print("--> [Router] Matched goodbye intent via Layer 1 regex.")
        matched_language = None
        if language in GOODBYE_REGEX_BY_LANG and GOODBYE_REGEX_BY_LANG[language].match(cleaned):
            matched_language = language
        elif GOODBYE_REGEX_BY_LANG["English"].match(cleaned):
            matched_language = "English"
        else:
            for lang, r_pattern in GOODBYE_REGEX_BY_LANG.items():
                if r_pattern.match(cleaned):
                    matched_language = lang
                    break
        if matched_language:
            language = matched_language
            safe_print(f"--> [Router] Override language to {language} based on goodbye pattern match.")
        return {
            "answer": get_direct_goodbye(language),
            "resources": [],
            "language": None,
            "emotion": None,
            "intent": "goodbye"
        }
    
    if GRATITUDE_REGEX.match(cleaned) and len(cleaned.split()) <= 3:
        safe_print("--> [Router] Matched gratitude intent via Layer 1 regex.")
        matched_language = None
        if language in GRATITUDE_REGEX_BY_LANG and GRATITUDE_REGEX_BY_LANG[language].match(cleaned):
            matched_language = language
        elif GRATITUDE_REGEX_BY_LANG["English"].match(cleaned):
            matched_language = "English"
        else:
            for lang, r_pattern in GRATITUDE_REGEX_BY_LANG.items():
                if r_pattern.match(cleaned):
                    matched_language = lang
                    break
        if matched_language:
            language = matched_language
            safe_print(f"--> [Router] Override language to {language} based on gratitude pattern match.")
        return {
            "answer": get_direct_gratitude(language),
            "resources": [],
            "language": None,
            "emotion": None,
            "intent": "gratitude"
        }

    # =============================================
    # LAYER 2: Parallel Intent & Emotion Classification
    # =============================================
    intent_task = asyncio.to_thread(classify_intent, query, language)
    emotion_task = asyncio.to_thread(classify_emotion, query)
    
    try:
        intent, emotions = await asyncio.gather(intent_task, emotion_task)
    except Exception as e:
        safe_print(f"--> [Router Error] Parallel classification failed: {e}")
        intent = "asking_mental_health_question"
        emotions = ["Sadness"]

    safe_print(f"--> [Router] Layer 2 LLM Intent: {intent}")
    safe_print(f"--> [Router] Language Detected: {language}")
    safe_print(f"--> [Router] Emotions Classified: {emotions}")

    # Handle query translation if enabled and detected language is not English
    query_en = query
    if ENABLE_TRANSLATION and language != "English":
        safe_print(f"--> [Router] Translating query from {language} to English...")
        query_en = translate_to_english(query)
        safe_print(f"--> [Router] Translated query: '{query_en}'")

    # =============================================
    # ROUTING LOGIC
    # =============================================
    if intent == "crisis":
        # Critical crisis detected -> Return localized safety template response
        safe_print("--> [Router] Classified intent as CRISIS. Routing to localized safety template.")
        crisis_msg = CRITICAL_CRISIS_RESPONSES.get(language, CRITICAL_CRISIS_RESPONSES["English"])
        return {
            "answer": crisis_msg,
            "resources": [],
            "language": language,
            "emotion": None,
            "intent": "crisis"
        }

    elif intent == "asking_mental_health_question":
        # Mental Health Topic -> Run full RAG pipeline
        safe_print("--> [Router] Routing to full Mental Health grounding RAG pipeline...")
        system_prompt = build_system_prompt(emotions, language, query)
        
        # Run synchronous RAG query in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            rag_instance.query, 
            user_query=query, 
            system_prompt=system_prompt, 
            translated_query=query_en,
            history=history
        )
        return {
            "answer": result["answer"],
            "resources": result.get("resources", []),
            "language": language,
            "emotion": emotions,
            "intent": intent
        }
        
    elif intent in ["greeting", "goodbye", "gratitude"]:
        # Conversational intent caught by LLM (not regex) -> Generate warm dynamic response using LLM with history
        safe_print(f"--> [Router] Routing to conversational LLM ({intent}) via Layer 2 fallback.")
        
        answer = await asyncio.to_thread(
            rag_instance.query_general,
            user_query=query,
            history=history,
            language=language
        )
            
        return {
            "answer": answer,
            "resources": [],
            "language": language,
            "emotion": None,
            "intent": intent
        }
        
    else:  # out_of_scope
        # Off-Topic -> Polite redirect response in user's language
        safe_print("--> [Router] Routing to Out-of-Scope redirect response.")
        emotions = []
        answer = OUT_OF_SCOPE_RESPONSES.get(language, OUT_OF_SCOPE_RESPONSES["English"])
        return {
            "answer": answer,
            "resources": [],
            "language": language,
            "emotion": None,
            "intent": intent
        }


async def preload_models():
    """
    Preloads all ML models (language detector, emotion classifier, intent classifier, and translator)
    concurrently on startup to avoid request-time latency spikes.
    """
    import asyncio
    from .modules import detect_language, classify_emotion, classify_intent
    
    # Define tasks for each model loading
    async def load_detector():
        try:
            # Triggers lazy initialization
            await asyncio.to_thread(detect_language, "hello")
            safe_print("--> Language detector model preloaded successfully.")
        except Exception as e:
            safe_print(f"Failed to preload language detector: {e}")

    async def load_emotion():
        try:
            # Triggers lazy initialization
            await asyncio.to_thread(classify_emotion, "hello")
            safe_print("--> Emotion classifier model preloaded successfully.")
        except Exception as e:
            safe_print(f"Failed to preload emotion classifier: {e}")

    async def load_intent():
        try:
            # Triggers lazy initialization
            await asyncio.to_thread(classify_intent, "hello")
            safe_print("--> Intent classifier model preloaded successfully.")
        except Exception as e:
            safe_print(f"Failed to preload intent classifier: {e}")

    async def load_translator():
        try:
            # Triggers lazy initialization
            if ENABLE_TRANSLATION:
                await asyncio.to_thread(translate_to_english, "hola")
                safe_print("--> Translator pipeline preloaded successfully.")
            else:
                safe_print("--> Translator preloading skipped (ENABLE_TRANSLATION=False).")
        except Exception as e:
            safe_print(f"Failed to preload translator: {e}")

    safe_print("--> Starting asynchronous preloading of classifiers and translator...")
    await asyncio.gather(
        load_detector(),
        load_emotion(),
        load_intent(),
        load_translator()
    )

"""
================================================================================
SERENE AI — CONVERSATIONAL ROUTING ENGINE
================================================================================
Routes queries dynamically using conversational fast-paths (regex checking),
crisis safety bypass triggers, and fallbacks to full semantic RAG logic.
================================================================================
"""

import asyncio
from langsmith import traceable
import os
from pathlib import Path
import re

# ------------------------------------------------------------------------------
# 1. Environment Loading & Configuration
# ------------------------------------------------------------------------------
from .config import config

# Read toggle for query translation
ENABLE_TRANSLATION = config.ENABLE_TRANSLATION

# Local project imports
from .modules import detect_language, classify_emotion, classify_intent
from .modules.multilingual_patterns import (
    GREETING_REGEX, GOODBYE_REGEX, GRATITUDE_REGEX,
    GREETING_REGEX_BY_LANG, GOODBYE_REGEX_BY_LANG, GRATITUDE_REGEX_BY_LANG,
    GREETING_RESPONSES, GOODBYE_RESPONSES, GRATITUDE_RESPONSES,
    CRITICAL_CRISIS_RESPONSES, OUT_OF_SCOPE_RESPONSES
)
from .modules.rag import detect_crisis

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
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            safe_print("--> [Translator Setup] Initializing local multilingual translation model...")
            tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
            model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
            _translator_pipeline = (tokenizer, model)
        
        tokenizer, model = _translator_pipeline
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
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

@traceable(name="route_query", run_type="chain")
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
    # 0.5. PROMPT INJECTION GUARDRAIL (takes < 1ms)
    # =============================================
    from .modules.rag import detect_prompt_injection
    if detect_prompt_injection(query):
        safe_print("--> [Router Guardrail] Matched prompt injection indicator. Declining query.")
        return {
            "answer": OUT_OF_SCOPE_RESPONSES.get(language, OUT_OF_SCOPE_RESPONSES["English"]),
            "resources": [],
            "language": language,
            "emotion": None,
            "intent": "out_of_scope"
        }

    # =============================================
    # 0.6. MEDICAL ADVICE BYPASS GUARDRAIL (takes < 1ms)
    # =============================================
    from .modules.rag import detect_medicine_query
    if detect_medicine_query(query):
        safe_print("--> [Router Guardrail] Matched medicine query keyword. Bypassing RAG and returning medical disclaimer.")
        from .modules.multilingual_patterns import MEDICAL_DISCLAIMERS
        return {
            "answer": MEDICAL_DISCLAIMERS.get(language, MEDICAL_DISCLAIMERS["English"]),
            "resources": [],
            "language": language,
            "emotion": None,
            "intent": "out_of_scope"
        }

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
        # Run synchronous RAG query in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            rag_instance.query, 
            user_query=query, 
            translated_query=query_en,
            history=history,
            emotions=emotions,
            language=language
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

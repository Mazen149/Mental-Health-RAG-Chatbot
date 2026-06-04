import os
import re
import asyncio
from .modules import detect_language, classify_emotion, classify_intent
from .rag_pipeline import build_system_prompt

# Read toggle for query translation (default is False, so we retrieve directly)
ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "False").lower() in ("true", "1", "yes")

def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            print(msg.encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

def translate_to_english(text: str, client) -> str:
    """
    Translates non-English text to English using Groq LLM.
    """
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise translator. Translate the user query into English. Output ONLY the English translation. Do not write any conversational filler, notes, or markdown."},
                {"role": "user", "content": text}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.0,
            max_tokens=150
        )
        translated = response.choices[0].message.content.strip()
        if translated:
            return translated
        return text
    except Exception as e:
        safe_print(f"Error in translating query to English: {e}")
        return text


GREETING_REGEX = re.compile(
    r"^(hi|hello|hey|greetings|good\s*(morning|afternoon|evening)|مرحبا|أهلا|أهلاً|السلام\s*عليكم|سلام\s*عليكم|bonjour|hola|salut|ciao|yo|hiya|welcome)(\s+.*)?$", 
    re.IGNORECASE
)

GOODBYE_REGEX = re.compile(
    r"^(bye|goodbye|see\s+you|take\s+care|مع\s*السلامة|إلى\s*اللقاء|وداعا|وداعاً|au\s+revoir|adiós|adios|nos\s+vemos|ciao|adieu)(\s+.*)?$",
    re.IGNORECASE
)

GRATITUDE_REGEX = re.compile(
    r"^(thanks?|thank\s+you|thx|شكر|شكراً|جزاك|تسلم|merci|gracias|danke|obrigado)(\s+.*)?$",
    re.IGNORECASE
)

def get_direct_greeting(query: str) -> str:
    """
    Returns a direct friendly greeting response in the user's language.
    """
    # Detect language using character set checks for fast response
    lower_query = query.lower()
    if any(char in lower_query for char in "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"):
        return "مرحباً! أنا هنا لدعمك ومساعدتك. كيف يمكنني مساعدتك اليوم؟"
    if "bonjour" in lower_query or "salut" in lower_query or "comment ça va" in lower_query:
        return "Bonjour ! Je suis là pour vous soutenir. Comment puis-je vous aider aujourd'hui ?"
    if "hola" in lower_query or "buenos dias" in lower_query:
        return "¡Hola! Estoy aquí para apoyarte. ¿Cómo puedo ayudarte hoy?"
    if "ciao" in lower_query:
        return "Ciao! Sono qui per supportarti. Come posso aiutarti oggi?"
    return "Hello! I am here to support you. How can I help you today?"

def get_direct_goodbye(query: str) -> str:
    """
    Returns a direct friendly goodbye response in the user's language.
    """
    lower_query = query.lower()
    if any(char in lower_query for char in "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"):
        return "مع السلامة! أتمنى لك يوماً هادئاً ومليئاً بالسكينة. أنا هنا دائماً إذا احتجت للدعم."
    if "au revoir" in lower_query or "adieu" in lower_query:
        return "Au revoir ! Prenez soin de vous. Je suis là si vous avez besoin de soutien."
    if "adiós" in lower_query or "adios" in lower_query or "nos vemos" in lower_query:
        return "¡Adiós! Cuídate mucho. Estoy aquí si necesitas apoyo."
    if "ciao" in lower_query:
        return "Ciao! Prenditi cura di te. Sono qui se hai bisogno di supporto."
    return "Goodbye! Take care of yourself. I am always here if you need support."

def get_direct_gratitude(query: str) -> str:
    """
    Returns a direct warm gratitude acknowledgement in the user's language.
    """
    lower_query = query.lower()
    if any(char in lower_query for char in "أبتثجحخدذرزسشصضطظعغفقكلمنهوي"):
        return "العفو! يسعدني أنني استطعت مساعدتك. لا تتردد في العودة في أي وقت تحتاج فيه إلى الدعم."
    if "merci" in lower_query:
        return "De rien ! Je suis heureux d'avoir pu vous aider. N'hésitez pas à revenir si vous avez besoin de soutien."
    if "gracias" in lower_query:
        return "¡De nada! Me alegra haber podido ayudarte. No dudes en volver si necesitas apoyo."
    if "danke" in lower_query:
        return "Gern geschehen! Es freut mich, dass ich helfen konnte. Kommen Sie jederzeit wieder."
    return "You're welcome! I'm glad I could help. Don't hesitate to come back anytime you need support."

async def route_query(query: str, rag_instance) -> dict:
    """
    Routes the user query dynamically in an asynchronous manner.
    
    Two-layer conversational intent detection:
      Layer 1: Regex fast-path for greeting/goodbye/gratitude (instant, 0ms)
      Layer 2: LLM classify_intent fallback for longer/complex messages
    
    INPUT: user query string
    OUTPUT: dict with answer, resources, and optional metadata fields
    """
    # =============================================
    # LAYER 1: Regex Fast-Path for Conversational Intents
    # =============================================
    cleaned = re.sub(r"[^\w\s\u0621-\u064A]", "", query).strip()
    
    if GREETING_REGEX.match(cleaned):
        safe_print(f"DEBUG: [Layer 1 Regex] Matched greeting")
        return {
            "answer": get_direct_greeting(query),
            "resources": [],
            "language": None,
            "emotion": None,
            "intent": "greeting"
        }
    
    if GOODBYE_REGEX.match(cleaned):
        safe_print(f"DEBUG: [Layer 1 Regex] Matched goodbye")
        return {
            "answer": get_direct_goodbye(query),
            "resources": [],
            "language": None,
            "emotion": None,
            "intent": "goodbye"
        }
    
    if GRATITUDE_REGEX.match(cleaned):
        safe_print(f"DEBUG: [Layer 1 Regex] Matched gratitude")
        return {
            "answer": get_direct_gratitude(query),
            "resources": [],
            "language": None,
            "emotion": None,
            "intent": "gratitude"
        }

    # =============================================
    # LAYER 2: LLM Intent Classification (parallel with lang + emotion)
    # =============================================
    intent_task = asyncio.to_thread(classify_intent, query)
    lang_task = asyncio.to_thread(detect_language, query)
    emotion_task = asyncio.to_thread(classify_emotion, query)
    
    try:
        intent, language, emotions = await asyncio.gather(intent_task, lang_task, emotion_task)
    except Exception as e:
        safe_print(f"Error in parallel classification: {e}")
        intent = "asking_mental_health_question"
        language = "English"
        emotions = ["Sadness"]

    safe_print(f"DEBUG: [Layer 2 LLM] Classified intent: {intent}")
    safe_print(f"DEBUG: Language detection -> Detected: {language}")
    safe_print(f"DEBUG: Emotion classification -> {emotions}")

    # Handle query translation if enabled and detected language is not English
    query_en = query
    if ENABLE_TRANSLATION and language != "English":
        safe_print(f"DEBUG: Translating query from {language} to English...")
        query_en = translate_to_english(query, rag_instance.client)
        safe_print(f"DEBUG: Translated query: '{query_en}'")

    # =============================================
    # ROUTING LOGIC
    # =============================================
    if intent == "asking_mental_health_question":
        # Mental Health Topic -> Run full RAG pipeline
        safe_print("--> Routing to Mental Health Topic pipeline...")
        system_prompt = build_system_prompt(emotions, language, query)
        
        # Run synchronous RAG query in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(rag_instance.query, query_en, system_prompt)
        return {
            "answer": result["answer"],
            "resources": result.get("resources", []),
            "language": language,
            "emotion": emotions,
            "intent": intent
        }
        
    elif intent in ["greeting", "goodbye", "gratitude"]:
        # Conversational intent caught by LLM (not regex) -> Use template responses
        safe_print(f"--> Routing to Conversational support ({intent}) via LLM detection...")
        
        if intent == "greeting":
            answer = get_direct_greeting(query)
        elif intent == "goodbye":
            answer = get_direct_goodbye(query)
        else:  # gratitude
            answer = get_direct_gratitude(query)
            
        return {
            "answer": answer,
            "resources": [],
            "language": language,
            "emotion": emotions,
            "intent": intent
        }
        
    else:  # out_of_scope
        # Off-Topic -> Polite redirect response in user's language
        safe_print("--> Routing to Out-of-Scope redirect response...")
        emotions = []
        redirect_prompts = {
            "Arabic": "أنا هنا لمساعدتك في الأسئلة المتعلقة بالصحة النفسية والعاطفية فقط. هل هناك أي موضوع متعلق بسلامتك النفسية ترغب في التحدث عنه؟",
            "French": "Je suis un assistant dédié au soutien en santé mentale. Je ne peux répondre qu'aux questions liées au bien-être psychologique. Comment puis-je vous aider dans ce domaine ?",
            "Spanish": "Soy un asistente de apoyo para la salud mental. Solo puedo responder a preguntas relacionadas con el bienestar emocional. ¿Cómo puedo ayudarte hoy en este ámbito?",
            "German": "Ich bin ein Assistent für mentale Gesundheit. Ich kann nur Fragen beantworten, die sich auf das emotionale Wohlbefinden beziehen. Wie kann ich Sie heute in diesem Bereich unterstützen?",
            "English": "I am a dedicated mental health support assistant. I can only assist with questions related to emotional and psychological well-being. How can I support you in that area today?"
        }
        answer = redirect_prompts.get(language, redirect_prompts["English"])
        return {
            "answer": answer,
            "resources": [],
            "language": language,
            "emotion": emotions,
            "intent": intent
        }

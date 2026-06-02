"""
Intent Classifier Module — 3-Stage Pipeline
=============================================
Routes each user message to one of five intents using a cascading pipeline:

    Stage 1: Rule-Based Keyword Matching   (~0 ms,  zero API calls)
    Stage 2: Multilingual Embedding Model  (~50 ms, local inference only)
    Stage 3: LLM Fallback via Groq API     (~600 ms, API call — last resort)

Intents:
    - greeting                       → Reply directly, no RAG
    - goodbye                        → Reply directly, no RAG
    - gratitude                      → Reply directly, no RAG
    - asking_mental_health_question  → Trigger RAG pipeline
    - out_of_scope                   → Politely decline

Supported languages (20):
    English, Spanish, French, Arabic, Hindi, Japanese, Chinese, Russian,
    Portuguese, Turkish, Italian, German, Greek, Dutch, Vietnamese,
    Swahili, Urdu, Polish, Bulgarian, Thai

Usage:
    from intent_classifier import IntentClassifier

    classifier = IntentClassifier(groq_api_key="gsk_...")
    result = classifier.classify("I've been feeling very anxious lately")
    # result -> {
    #     "intent": "asking_mental_health_question",
    #     "confidence": 0.95,
    #     "stage": "embedding",
    #     ...
    # }

    # Or use the router for production integration:
    router = IntentRouter(groq_api_key="gsk_...")
    route = router.route("Hello!", detected_language="en")
"""

import os
import re
import json
import time
from typing import Optional
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Dynamically locate the project root relative to this file
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parents[2]  # src/chatbot/nlp -> three levels up to project github root
_ENV_PATH = _PROJECT_ROOT / ".env"
# Explicitly load from the absolute path of your project root's .env
load_dotenv(dotenv_path=_ENV_PATH)

MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_THRESHOLD = 0.7

VALID_INTENTS = [
    "greeting",
    "goodbye",
    "gratitude",
    "asking_mental_health_question",
    "out_of_scope",
]

API_CALL_DELAY = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Rules (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_RULES = {
    "greeting": {
        "en": ["hi", "hello", "hey", "howdy", "greetings", "good morning",
               "good afternoon", "good evening", "good day", "sup",
               "what's up", "whats up", "yo", "hiya", "ello", "good to see you"],
        "es": ["hola", "buenos días", "buenos dias", "buenas tardes",
               "buenas noches", "buenas", "qué tal", "que tal", "buen día"],
        "fr": ["bonjour", "bonsoir", "salut", "coucou", "bonne journée",
               "bonne matinée"],
        "ar": ["مرحبا", "مرحباً", "السلام عليكم", "أهلا", "أهلاً",
               "صباح الخير", "مساء الخير", "هلا", "أهلاً وسهلاً"],
        "hi": ["नमस्ते", "नमस्कार", "हेलो", "प्रणाम", "राम राम",
               "सुप्रभात", "शुभ संध्या"],
        "ja": ["こんにちは", "こんばんは", "おはよう", "おはようございます",
               "やあ", "ハロー", "よう", "どうも"],
        "zh": ["你好", "您好", "嗨", "早上好", "早", "哈喽", "大家好",
               "晚上好", "下午好"],
        "ru": ["привет", "здравствуйте", "здравствуй", "добрый день",
               "доброе утро", "добрый вечер", "хай", "хей", "приветствую"],
        "pt": ["olá", "oi", "bom dia", "boa tarde", "boa noite", "alô",
               "salve", "e aí"],
        "tr": ["merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar",
               "selamlar", "n'aber"],
        "it": ["ciao", "buongiorno", "buonasera", "salve", "buon pomeriggio"],
        "de": ["hallo", "guten morgen", "guten tag", "guten abend", "servus",
               "moin", "tach"],
        "el": ["γεια", "γεια σου", "γεια σας", "καλημέρα", "καλησπέρα",
               "χαίρετε"],
        "nl": ["hallo", "hoi", "goedendag", "goedemorgen", "goedemiddag",
               "goedenavond", "hee"],
        "vi": ["xin chào", "chào", "chào buổi sáng", "chào buổi chiều",
               "chào mừng", "chào bạn"],
        "sw": ["habari", "jambo", "hujambo", "mambo", "salam", "shikamoo",
               "karibu", "habari yako"],
        "ur": ["ہیلو", "السلام علیکم", "آداب", "سلام", "ادب",
               "صبح بخیر", "شام بخیر"],
        "pl": ["cześć", "dzień dobry", "hej", "siema", "witaj", "witajcie",
               "dobry wieczór"],
        "bg": ["здравей", "здравейте", "добро утро", "добър ден",
               "добър вечер", "хей"],
        "th": ["สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี",
               "ดีจ้า", "ดีครับ", "ดีค่ะ"],
    },
    "goodbye": {
        "en": ["bye", "goodbye", "farewell", "see you", "see ya", "take care",
               "later", "gotta go", "signing off", "good night", "ttyl", "cya",
               "have a good one", "until next time", "so long", "tata",
               "ta-ta", "cheerio", "catch you later", "talk later"],
        "es": ["adiós", "adios", "hasta luego", "nos vemos", "chao", "chau",
               "hasta pronto", "hasta mañana", "que te vaya bien"],
        "fr": ["au revoir", "à bientôt", "adieu", "bonne nuit", "à plus",
               "à plus tard", "à la prochaine"],
        "ar": ["مع السلامة", "وداعاً", "وداعا", "إلى اللقاء", "باي",
               "تصبح على خير", "الى اللقاء"],
        "hi": ["अलविदा", "फिर मिलेंगे", "टाटा", "बाय", "शुभ रात्रि"],
        "ja": ["さようなら", "バイバイ", "またね", "じゃあね", "お休み",
               "またあした", "では", "失礼します"],
        "zh": ["再见", "拜拜", "再会", "晚安", "回头见", "下次见",
               "先走了", "再联系"],
        "ru": ["до свидания", "пока", "прощай", "до встречи",
               "всего хорошего", "спокойной ночи", "увидимся"],
        "pt": ["tchau", "adeus", "até logo", "até mais", "boa noite",
               "até amanhã", "até breve"],
        "tr": ["güle güle", "hoşçakal", "görüşürüz", "iyi geceler",
               "bay bay", "görüşmek üzere"],
        "it": ["arrivederci", "addio", "a presto", "buonanotte",
               "ci vediamo", "a domani"],
        "de": ["auf wiedersehen", "tschüss", "tschüssi", "bis bald",
               "gute nacht", "bis dann", "bis später"],
        "el": ["αντίο", "τα λέμε", "αντίο σας", "καληνύχτα",
               "εις το επανιδείν"],
        "nl": ["tot ziens", "doei", "dag", "tot later", "goedenacht",
               "tot snel", "tot morgen"],
        "vi": ["tạm biệt", "hẹn gặp lại", "chào tạm biệt",
               "chúc ngủ ngon", "bái bai"],
        "sw": ["kwa heri", "kwaheri", "tutaonana", "lala salama",
               "usiku mwema"],
        "ur": ["خدا حافظ", "اللہ حافظ", "الوداع", "پھر ملیں گے",
               "بائے", "شب بخیر"],
        "pl": ["do widzenia", "pa", "na razie", "dobranoc",
               "do zobaczenia"],
        "bg": ["довиждане", "чао", "до скоро", "лека нощ",
               "ще се видим"],
        "th": ["ลาก่อน", "บาย", "แล้วเจอกัน", "ราตรีสวัสดิ์", "ฝันดี"],
    },
    "gratitude": {
        "en": ["thanks", "thank you", "thank u", "thx", "ty", "cheers",
               "grateful", "i appreciate", "much appreciated", "many thanks",
               "ta", "thanks a lot", "thank you very much", "so thankful"],
        "es": ["gracias", "muchas gracias", "mil gracias",
               "te lo agradezco", "muy agradecido", "muy agradecida"],
        "fr": ["merci", "merci beaucoup", "je vous remercie",
               "je te remercie", "grand merci", "merci infiniment"],
        "ar": ["شكرا", "شكراً", "شكرا جزيلا", "ممنون", "متشكر",
               "جزاك الله خيراً", "الله يعطيك العافية", "مشكور"],
        "hi": ["धन्यवाद", "शुक्रिया", "थैंक यू", "आभार",
               "बहुत बहुत धन्यवाद"],
        "ja": ["ありがとう", "ありがとうございます", "どうもありがとう",
               "感謝します", "ありがとうございました"],
        "zh": ["谢谢", "谢谢你", "谢谢您", "多谢", "非常感谢", "感谢",
               "太感谢了"],
        "ru": ["спасибо", "благодарю", "большое спасибо",
               "благодарен", "благодарна"],
        "pt": ["obrigado", "obrigada", "muito obrigado", "muito obrigada",
               "agradecido", "agradecida"],
        "tr": ["teşekkürler", "teşekkür ederim", "sağ ol",
               "çok teşekkürler", "minnettarım"],
        "it": ["grazie", "grazie mille", "ti ringrazio", "vi ringrazio",
               "mille grazie", "grazie tante"],
        "de": ["danke", "danke schön", "vielen dank", "herzlichen dank",
               "dankeschön", "besten dank"],
        "el": ["ευχαριστώ", "ευχαριστώ πολύ", "σε ευχαριστώ",
               "σας ευχαριστώ"],
        "nl": ["dank je", "dank u", "dank je wel", "bedankt",
               "hartelijk dank", "ontzettend bedankt"],
        "vi": ["cảm ơn", "cảm ơn bạn", "cảm ơn nhiều",
               "xin cảm ơn", "cảm ơn rất nhiều"],
        "sw": ["asante", "asante sana", "nashukuru", "shukrani",
               "asante kwa msaada"],
        "ur": ["شکریہ", "بہت شکریہ", "ممنون", "آپ کا شکریہ"],
        "pl": ["dziękuję", "dziękuję bardzo", "dzięki",
               "wielkie dzięki", "bardzo dziękuję"],
        "bg": ["благодаря", "благодаря много", "благодарен съм",
               "мерси", "благодарна съм"],
        "th": ["ขอบคุณ", "ขอบคุณมาก", "ขอบใจ", "ขอบพระคุณ",
               "ขอบคุณครับ", "ขอบคุณค่ะ"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Mental Health Signals (Stage 1 false-positive guard)
# ─────────────────────────────────────────────────────────────────────────────

MENTAL_HEALTH_SIGNALS = {
    # English
    "anxiety", "anxious", "depressed", "depression", "stress", "stressed",
    "panic", "therapy", "therapist", "mental health", "suicidal", "self-harm",
    "burnout", "lonely", "loneliness", "grief", "trauma", "overwhelmed",
    "hopeless", "empty", "exhausted", "sad", "fear", "worried", "worry",
    "crying", "helpless", "worthless", "numb", "ptsd", "ocd", "adhd",
    "mood", "bipolar", "psychologist", "psychiatrist", "counseling",
    "counselor", "insomnia", "sleeping problems", "eating disorder",
    # Spanish
    "ansiedad", "depresión", "depresion", "estresado", "estrés", "estres",
    "terapia", "salud mental", "pánico", "panico", "angustia", "soledad",
    "tristeza", "agotado", "desesperado", "psicólogo", "burnout",
    # French
    "anxiété", "dépression", "stress", "thérapie", "panique",
    "santé mentale", "épuisement", "tristesse", "seul", "désespoir",
    "peur", "déprimé", "psychologue", "angoisse",
    # Arabic
    "قلق", "اكتئاب", "ضغط", "علاج نفسي", "نفسي", "حزن", "وحدة",
    "يأس", "خوف", "بكاء", "انهيار", "مشكلة نفسية", "معالج",
    # Hindi
    "चिंता", "तनाव", "अवसाद", "मानसिक", "घबराहट", "डर", "अकेलापन",
    "रोना", "निराशा", "मनोचिकित्सक",
    # Japanese
    "不安", "うつ", "ストレス", "療法", "パニック", "メンタル",
    "孤独", "恐怖", "疲労", "睡眠", "カウンセリング", "自傷",
    # Chinese
    "焦虑", "抑郁", "压力", "治疗", "恐慌", "心理", "恐惧",
    "疲惫", "失眠", "心理咨询", "创伤",
    # Russian
    "тревога", "депрессия", "стресс", "терапия", "паника", "психическ",
    "одинок", "страх", "усталость", "бессонница", "психолог", "горе",
    # Portuguese
    "ansiedade", "depressão", "depressao", "estresse", "terapia", "pânico",
    "panico", "saúde mental", "solidão", "medo", "tristeza",
    "exausto", "desespero", "psicólogo",
    # Turkish
    "kaygı", "kaygi", "depresyon", "stres", "terapi", "panik",
    "ruh sağlığı", "yalnız", "korku", "yorgunluk", "uykusuzluk",
    # Italian
    "ansia", "depressione", "terapia", "panico", "salute mentale",
    "solitudine", "paura", "esaurito", "disperato", "psicologo",
    # German
    "angst", "depression", "therapie", "panik",
    "einsamkeit", "trauer", "erschöpft", "schlaflosigkeit", "psychologe",
    # Greek
    "άγχος", "κατάθλιψη", "στρες", "θεραπεία", "μοναξιά",
    "φόβος", "εξάντληση", "ψυχολόγος",
    # Dutch
    "depressie", "therapie", "paniek", "eenzaamheid",
    "verdriet", "uitgeput", "psycholoog", "slaapproblemen",
    # Vietnamese
    "lo lắng", "trầm cảm", "căng thẳng", "trị liệu", "cô đơn",
    "sợ hãi", "kiệt sức", "tâm lý",
    # Swahili
    "wasiwasi", "msongo", "huzuni", "upweke", "hofu", "uchovu",
    "afya ya akili",
    # Urdu
    "پریشانی", "ذہنی", "تناؤ", "ڈپریشن", "اکیلاپن", "خوف",
    "تھکاوٹ", "نیند", "نفسیات",
    # Polish
    "lęk", "depresja", "stres", "terapia", "panika", "samotność",
    "smutek", "wyczerpanie", "psycholog",
    # Bulgarian
    "тревожност", "депресия", "стрес", "терапия", "паника", "самота",
    "страх", "умора", "психолог", "безсъние",
    # Thai
    "วิตกกังวล", "ซึมเศร้า", "ความเครียด", "บำบัด", "โดดเดี่ยว",
    "กลัว", "เหนื่อย", "นักจิตวิทยา",
}


# ─────────────────────────────────────────────────────────────────────────────
# Intent Exemplars (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_EXEMPLARS = {
    "greeting": [
        "Hello there!", "Hi, how are you?", "Good morning!", "Hey, nice to meet you",
        "Greetings!", "Hey! Good to see you", "Hi, I'm new here",
        "Hola", "Bonjour", "مرحبا", "नमस्ते", "こんにちは",
        "你好", "您好", "嗨", "早上好", "早", "哈喽", "大家好",
        "晚上好", "下午好", "Привет", "Merhaba", "Ciao", "Hallo",
        "Γεια σου", "สวัสดีครับ", "Habari",
    ],
    "goodbye": [
        "Goodbye!", "See you later!", "Take care!", "Bye bye", "I have to go now",
        "Until next time", "Farewell, and thanks", "Goodnight everyone",
        "Adiós", "Au revoir", "مع السلامة", "再见", "До свидания",
        "Tschüss", "Arrivederci", "Kwa heri", "さようなら", "Até logo",
        "Do widzenia", "ลาก่อน",
    ],
    "gratitude": [
        "Thank you so much!", "Thanks a lot!", "I really appreciate it",
        "You've been very helpful, thanks!", "I'm grateful for your support",
        "Thanks, that was very useful", "I appreciate your help enormously",
        "Gracias", "Merci beaucoup", "شكرا", "谢谢", "Спасибо",
        "Danke schön", "Grazie mille", "ありがとうございます",
        "Dziękuję bardzo", "ขอบคุณมาก", "Asante sana",
    ],
    "asking_mental_health_question": [
        "I've been feeling very anxious lately, what should I do?",
        "How do I deal with depression?",
        "I'm struggling with panic attacks",
        "Can you help me understand my stress?",
        "I feel hopeless and don't know what to do",
        "What therapy techniques help with anxiety?",
        "I've been having suicidal thoughts",
        "I feel burned out and exhausted all the time",
        "I'm so lonely and nobody understands me",
        "Is it normal to feel empty and numb?",
        "I can't sleep because of my anxiety",
        "Me siento muy deprimido, ¿qué hago?",
        "Je souffre d'anxiété sévère",
        "أشعر بالاكتئاب الشديد",
        "我感到非常焦虑，晚上睡不好",
        "У меня депрессия уже несколько месяцев",
        "Ich habe Angst und weiß nicht was ich tun soll",
        "Mam ataki paniki w pracy",
        "Tükenmiş hissediyorum og bunun üstesinden gelemiyorum",
        "Continuo a sentirmi ansioso, cosa posso fare?",
        "Hi, I've been struggling with depression a lot lately",
    ],
    "out_of_scope": [
        "What is the capital of France?",
        "How do I code in Python?",
        "What's the weather today?",
        "Tell me a joke",
        "What are the best stocks to invest in?",
        "Can you help me with my homework?",
        "今天天气怎么样",
        "¿Cuál es la capital de España?",
        "Quel est le prix du bitcoin?",
        "asdkjhaskdjh",
        "2+2=?",
        "SELECT * FROM users",
        "Book me a flight to London",
        "Who is the president of France?",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM System Prompt (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
<task>
You are an intent classification engine for a multilingual mental health support chatbot.
Your role is classification only — do NOT answer the user's question.
</task>

<intents>
<intent name="greeting">Hello, hi, good morning, or any opening salutation.</intent>
<intent name="goodbye">Bye, farewell, see you, take care, or closing the conversation.</intent>
<intent name="gratitude">Thanks or appreciation.</intent>
<intent name="asking_mental_health_question">
Any question, statement, or expression relating to mental health, including:
anxiety, depression, stress, panic attacks, therapy, counseling, suicidal thoughts,
self-harm, emotional support, coping strategies, grief, trauma, burnout, loneliness,
emotional numbness, hopelessness, sleep problems caused by mental health, eating disorders.
</intent>
<intent name="out_of_scope">Anything unrelated to mental health, or completely unclear/gibberish.</intent>
</intents>

<rules>
1. If a message contains BOTH a greeting AND a mental health question → asking_mental_health_question.
2. If a message contains BOTH gratitude AND goodbye → goodbye.
3. If confidence is genuinely low (< 0.4) → default to out_of_scope.
4. The message may be in any language or script — classify by meaning, not language.
5. Think step-by-step internally; output ONLY the JSON object, nothing else.
</rules>

<response_format>
Return a single-line JSON object with exactly two keys:
  {"intent": "<one of the 5 intents>", "confidence": <float 0.0-1.0>}
Example: {"intent": "greeting", "confidence": 0.92}
</response_format>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Routing Table
# ─────────────────────────────────────────────────────────────────────────────

ROUTING_TABLE = {
    "greeting": {
        "action":   "direct_reply",
        "use_rag":  False,
        "template": "Hello! I'm here to support you with mental health topics. "
                    "Feel free to share what's on your mind. 😊",
    },
    "goodbye": {
        "action":   "direct_reply",
        "use_rag":  False,
        "template": "Take care of yourself! Remember, I'm here whenever you need support. 💙",
    },
    "gratitude": {
        "action":   "direct_reply",
        "use_rag":  False,
        "template": "You're very welcome! I'm glad I could help. "
                    "Don't hesitate to reach out anytime. 🌟",
    },
    "asking_mental_health_question": {
        "action":   "rag_pipeline",
        "use_rag":  True,
        "template": None,
    },
    "out_of_scope": {
        "action":   "decline",
        "use_rag":  False,
        "template": "I'm specialised in mental health support, so I can't help with that topic. "
                    "If you have questions about anxiety, depression, stress, or emotional wellbeing, "
                    "I'm here for you! 💙",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Rule-Based Keyword Matching (helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _build_keyword_lookup() -> dict:
    """Flatten KEYWORD_RULES into keyword_lowercase → intent."""
    lookup = {}
    for intent, lang_dict in KEYWORD_RULES.items():
        for _lang, keywords in lang_dict.items():
            for kw in keywords:
                lookup[kw.lower()] = intent
    return lookup


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Match keyword using word-boundary for ASCII, substring for non-ASCII."""
    if all(ord(c) < 128 for c in keyword):
        pattern = r'(?<![a-z0-9])' + re.escape(keyword) + r'(?![a-z0-9])'
        return bool(re.search(pattern, text, re.IGNORECASE))
    else:
        return keyword in text


def _prune_overlapping_matches(matched: dict[str, list]) -> dict[str, list]:
    """Remove shorter keyword matches contained inside longer phrases."""
    if not matched:
        return matched

    all_matches = [(kw, intent) for intent, kws in matched.items() for kw in kws]
    all_matches.sort(key=lambda x: len(x[0]), reverse=True)

    kept: list[tuple[str, str]] = []
    for kw, intent in all_matches:
        kw_lower = kw.lower()
        if any(kw_lower != kept_kw and kw_lower in kept_kw for kept_kw, _ in kept):
            continue
        kept.append((kw_lower, intent))

    pruned: dict[str, list] = {}
    for kw, intent in kept:
        pruned.setdefault(intent, []).append(kw)

    return pruned


def _looks_like_code_or_noise(message: str) -> bool:
    """Detect code-like inputs, translation requests, or gibberish."""
    msg = message.strip()
    if not msg:
        return True

    msg_lower = msg.lower()
    if "translate" in msg_lower or "translation" in msg_lower:
        language_names = (
            "english", "spanish", "french", "arabic", "hindi", "japanese",
            "chinese", "russian", "portuguese", "turkish", "italian", "german",
            "greek", "dutch", "vietnamese", "swahili", "urdu", "polish",
            "bulgarian", "thai",
        )
        if any(lang in msg_lower for lang in language_names):
            return True

    code_regexes = [
        r"\bprint\s*\(",
        r"\bconsole\.log\s*\(",
        r"\bdef\s+\w+\s*\(",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\s+\w+",
        r"\bselect\b.+\bfrom\b",
    ]
    if any(re.search(rx, msg_lower) for rx in code_regexes):
        return True

    if re.search(r"[{}\[\]]", msg):
        return True

    # Gibberish heuristic
    if all(ord(c) < 128 for c in msg):
        tokens = re.findall(r"[a-zA-Z]+", msg)
        if len(tokens) == 1:
            token = tokens[0]
            if len(token) >= 12:
                vowel_ratio = sum(ch in "aeiouy" for ch in token.lower()) / len(token)
                if vowel_ratio < 0.25:
                    return True

    return False


def _has_additional_meaningful_content(message: str, matched_keywords: list) -> bool:
    """
    Check if message contains mental health signals or substantial content
    beyond matched keywords.
    """
    msg_lower = message.lower()

    # Check 1: mental health signal present
    for signal in MENTAL_HEALTH_SIGNALS:
        if signal in msg_lower:
            return True

    # Check 2: substantial residual after stripping matched keywords
    residual = msg_lower
    for kw in matched_keywords:
        residual = residual.replace(kw.lower(), " ")

    residual = re.sub(r'[^\w\s]', ' ', residual)
    residual = re.sub(r'\s+', ' ', residual).strip()

    meaningful_tokens = [t for t in residual.split() if len(t) > 2]
    if len(meaningful_tokens) >= 4:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Intent Classifier Class
# ─────────────────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    3-Stage Intent Classification Pipeline.

    Parameters
    ----------
    groq_api_key : str, optional
        Groq API key for Stage 3 LLM fallback. If not provided, reads from
        GROQ_API_KEY environment variable. Stage 3 will be skipped if unavailable.
    embedding_threshold : float
        Minimum cosine similarity for Stage 2 to commit (default: 0.7).
    llm_confidence_threshold : float
        Minimum LLM confidence; below this → out_of_scope (default: 0.4).
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        embedding_threshold: float = EMBEDDING_THRESHOLD,
        llm_confidence_threshold: float = 0.4,
    ):
        self.embedding_threshold = embedding_threshold
        self.llm_confidence_threshold = llm_confidence_threshold

        # Build keyword lookup
        self._keyword_lookup = _build_keyword_lookup()

        # Initialise embedding model (Stage 2)
        self._embedding_model = None
        self._intent_centroids = None
        self._init_embedding_model()

        # Initialise Groq client (Stage 3)
        self._groq_client = None
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
            except ImportError:
                print("Warning: 'groq' package not installed. Stage 3 (LLM) will be disabled.")
            except Exception as e:
                print(f"Warning: Failed to initialise Groq client: {e}. Stage 3 disabled.")

    def _init_embedding_model(self):
        """Load the multilingual embedding model and pre-compute intent centroids."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            # Pre-compute centroids
            all_exemplars = []
            intent_sizes = {}
            for intent, exemplars in INTENT_EXEMPLARS.items():
                intent_sizes[intent] = len(exemplars)
                all_exemplars.extend(exemplars)

            all_embeddings = self._embedding_model.encode(
                all_exemplars,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )

            self._intent_centroids = {}
            idx = 0
            for intent in intent_sizes:
                size = intent_sizes[intent]
                chunk = all_embeddings[idx: idx + size]
                centroid = np.mean(chunk, axis=0)
                centroid /= np.linalg.norm(centroid)
                self._intent_centroids[intent] = centroid
                idx += size

        except ImportError:
            print("Warning: 'sentence-transformers' not installed. Stage 2 (embedding) disabled.")
        except Exception as e:
            print(f"Warning: Failed to initialise embedding model: {e}. Stage 2 disabled.")

    # ── Stage 1: Rule-Based ──────────────────────────────────────────────

    def _rule_based_classify(self, message: str) -> Optional[dict]:
        """Stage 1: Rule-based keyword matching."""
        t0 = time.time()

        if _looks_like_code_or_noise(message):
            return {
                "intent": "out_of_scope",
                "confidence": 0.8,
                "stage": "rule_based",
                "matched_keywords": [],
                "latency_ms": int((time.time() - t0) * 1000),
            }

        msg_lower = message.lower().strip()

        matched: dict[str, list] = {}
        for keyword in sorted(self._keyword_lookup.keys(), key=len, reverse=True):
            intent = self._keyword_lookup[keyword]
            if _keyword_in_text(keyword, msg_lower):
                matched.setdefault(intent, []).append(keyword)

        if not matched:
            return None

        matched = _prune_overlapping_matches(matched)
        if not matched:
            return None

        all_matched = [kw for kws in matched.values() for kw in kws]

        # False-positive guard
        if _has_additional_meaningful_content(message, all_matched):
            return None

        # Priority resolution
        if "goodbye" in matched:
            resolved = "goodbye"
        elif "gratitude" in matched:
            resolved = "gratitude"
        elif "greeting" in matched:
            resolved = "greeting"
        else:
            resolved = next(iter(matched))

        return {
            "intent": resolved,
            "confidence": 0.95,
            "stage": "rule_based",
            "matched_keywords": matched.get(resolved, []),
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # ── Stage 2: Embedding ───────────────────────────────────────────────

    def _embedding_classify(self, message: str) -> Optional[dict]:
        """Stage 2: Embedding-based intent classification."""
        if self._embedding_model is None or self._intent_centroids is None:
            return None

        t0 = time.time()

        query_vec = self._embedding_model.encode(
            [message], normalize_embeddings=True, show_progress_bar=False
        )[0]

        similarities = {
            intent: float(np.dot(query_vec, centroid))
            for intent, centroid in self._intent_centroids.items()
        }

        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]

        latency_ms = int((time.time() - t0) * 1000)

        if best_score < self.embedding_threshold:
            return None

        return {
            "intent": best_intent,
            "confidence": round(best_score, 4),
            "stage": "embedding",
            "all_similarities": {k: round(v, 4) for k, v in similarities.items()},
            "latency_ms": latency_ms,
        }

    # ── Stage 3: LLM Fallback ────────────────────────────────────────────

    def _build_llm_messages(self, user_message: str) -> list:
        """Construct the messages list for the Groq API call."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    @staticmethod
    def _parse_llm_response(raw_text: str) -> dict:
        """Robustly parse the LLM's raw output into {intent, confidence, parse_error}."""
        raw_text = raw_text.strip()

        # Strategy 1: Direct JSON
        try:
            parsed = json.loads(raw_text)
            intent = parsed.get("intent", "").strip().lower()
            confidence = float(parsed.get("confidence", 0.5))
            if intent in VALID_INTENTS:
                return {"intent": intent, "confidence": confidence, "parse_error": False}
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Regex extraction
        im = re.search(r'"intent"\s*:\s*"([^"]+)"', raw_text, re.IGNORECASE)
        cm = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw_text, re.IGNORECASE)
        if im and im.group(1).strip().lower() in VALID_INTENTS:
            return {
                "intent": im.group(1).strip().lower(),
                "confidence": float(cm.group(1)) if cm else 0.5,
                "parse_error": True,
            }

        # Strategy 3: Keyword scan
        raw_lower = raw_text.lower()
        for label in VALID_INTENTS:
            if label in raw_lower:
                return {"intent": label, "confidence": 0.4, "parse_error": True}

        return {"intent": "out_of_scope", "confidence": 0.0, "parse_error": True}

    def _llm_classify(self, user_message: str, max_retries: int = 3) -> dict:
        """Stage 3: LLM-based intent classification via Groq."""
        if self._groq_client is None:
            return {
                "intent": "out_of_scope", "confidence": 0.0,
                "stage": "llm", "raw": "LLM_UNAVAILABLE",
                "parse_error": True, "latency_ms": -1,
            }

        messages = self._build_llm_messages(user_message)

        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                response = self._groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=60,
                    temperature=0.0,
                )
                latency_ms = int((time.time() - t0) * 1000)
                raw_output = response.choices[0].message.content.strip()
                result = self._parse_llm_response(raw_output)
                result["stage"] = "llm"
                result["raw"] = raw_output
                result["latency_ms"] = latency_ms
                return result

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)

        return {
            "intent": "out_of_scope", "confidence": 0.0,
            "stage": "llm", "raw": "API_FAILURE",
            "parse_error": True, "latency_ms": -1,
        }

    # ── Unified Pipeline ─────────────────────────────────────────────────

    def classify(self, message: str) -> dict:
        """
        Classify the intent of a user message using the 3-stage pipeline.

        Parameters
        ----------
        message : str
            Raw user input (any of 20 supported languages).

        Returns
        -------
        dict
            {
                "intent": str,       # one of VALID_INTENTS
                "confidence": float, # 0.0–1.0
                "stage": str,        # "rule_based" | "embedding" | "llm"
                "latency_ms": int,   # wall-clock time for the resolving stage
                ...                  # stage-specific keys
            }
        """
        # Stage 1: Rule-Based
        result = self._rule_based_classify(message)
        if result is not None:
            return result

        # Stage 2: Embedding
        result = self._embedding_classify(message)
        if result is not None:
            return result

        # Stage 3: LLM Fallback
        result = self._llm_classify(message)
        if result["confidence"] < self.llm_confidence_threshold:
            result["intent"] = "out_of_scope"

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Intent Router (production wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Production router wrapping the 3-stage pipeline.

    Integrates with:
        ← Module 1 (Language Detection) → feeds detected_language to route()
        ← Module 2 (Emotion Classifier) → runs after routing when use_rag=True
        → Module 4 (RAG Q&A)            → triggered when use_rag=True

    Parameters
    ----------
    groq_api_key : str, optional
        Groq API key. Falls back to GROQ_API_KEY env var.
    embedding_threshold : float
        Min cosine similarity for Stage 2 (default: 0.7).
    llm_confidence_threshold : float
        Min LLM confidence (default: 0.4).
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        embedding_threshold: float = EMBEDDING_THRESHOLD,
        llm_confidence_threshold: float = 0.4,
    ):
        self.classifier = IntentClassifier(
            groq_api_key=groq_api_key,
            embedding_threshold=embedding_threshold,
            llm_confidence_threshold=llm_confidence_threshold,
        )

    def route(self, user_message: str, detected_language: str = "en") -> dict:
        """
        Classify intent and return routing information.

        Parameters
        ----------
        user_message : str
            Raw user input.
        detected_language : str
            ISO 639-1 language code from the language detector.

        Returns
        -------
        dict
            {
                "user_message": str,
                "detected_language": str,
                "intent": str,
                "confidence": float,
                "stage": str,
                "action": str,          # "direct_reply" | "rag_pipeline" | "decline"
                "use_rag": bool,
                "response": str | None, # Template response or None (use RAG)
            }
        """
        classification = self.classifier.classify(user_message)
        intent = classification["intent"]
        route_info = ROUTING_TABLE[intent]

        return {
            "user_message": user_message,
            "detected_language": detected_language,
            "intent": intent,
            "confidence": classification["confidence"],
            "stage": classification["stage"],
            "action": route_info["action"],
            "use_rag": route_info["use_rag"],
            "response": route_info["template"],
        }

    def format_routing_log(self, r: dict) -> str:
        """Format a routing result as a human-readable log line."""
        return (
            f"[ROUTER] '{r['user_message']}' | "
            f"intent={r['intent']} ({r['confidence']:.2f}) via {r['stage']} | "
            f"action={r['action']} | use_rag={r['use_rag']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function (module-level)
# ─────────────────────────────────────────────────────────────────────────────

_classifier_instance = None


def classify_intent(message: str, groq_api_key: str | None = None) -> dict:
    """
    Convenience function: classify intent without manually creating an IntentClassifier.

    Uses a lazily-initialised singleton instance.
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier(groq_api_key=groq_api_key)
    return _classifier_instance.classify(message)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initialising Intent Classifier (3-stage pipeline)...")
    classifier = IntentClassifier()

    test_cases = [
        ("hi",                                               "greeting"),
        ("Hello!",                                           "greeting"),
        ("مرحبا",                                             "greeting"),
        ("bye",                                              "goodbye"),
        ("さようなら",                                        "goodbye"),
        ("thanks",                                           "gratitude"),
        ("ขอบคุณมาก",                                         "gratitude"),
        ("I've been feeling very anxious lately",            "asking_mental_health_question"),
        ("我最近压力很大，晚上睡不着",                          "asking_mental_health_question"),
        ("What is the capital of France?",                   "out_of_scope"),
        ("print('hello world')",                             "out_of_scope"),
    ]

    print("\n" + "=" * 90)
    print(" INTENT CLASSIFIER — Quick Test")
    print("=" * 90)

    passed = 0
    for msg, expected in test_cases:
        result = classifier.classify(msg)
        ok = result["intent"] == expected
        if ok:
            passed += 1
        icon = "✅" if ok else "❌"
        print(
            f"  {icon}  [{result['stage']:<10}] "
            f"{result['intent']:<35} "
            f"(conf={result['confidence']:.2f})  "
            f"← \"{msg}\""
        )
        if result["stage"] == "llm":
            time.sleep(API_CALL_DELAY)

    print(f"\nPassed: {passed}/{len(test_cases)}")
    print("=" * 90)

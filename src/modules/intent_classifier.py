import re
from typing import Callable, Literal

import dspy
from huggingface_hub import InferenceClient
import numpy as np
from pydantic import BaseModel, Field

# Locate project root and load environment
from ..config import config
from ..prompts.prompts import IntentClassifierModule


# =============================================================================
# FIGURATIVE / IDIOMATIC EXPRESSION GUARDRAIL
# =============================================================================
# This comprehensive list catches figurative, idiomatic, and slang expressions
# that contain words like "death", "kill", "die", etc. but are NOT crisis.
# These patterns are checked BEFORE any ML classifier runs.
# =============================================================================

# Compiled regex patterns for figurative expressions across languages
# Each pattern is designed to match non-literal uses of death/harm words
_FIGURATIVE_PATTERNS = [
    # ---- Arabic figurative / slang (Egyptian, Levantine, Gulf, Maghreb) ----
    # "بموت في X" / "بموت من X" = I'm crazy about X / dying of X (love, laughter, etc.)
    re.compile(r'بم[وؤ]*ت\s+(في|فى|من|على|عليك|عليكي|عليها|عليه|فيك|فيكي|فيها|فيه)', re.IGNORECASE),
    # "أموت في" / "أموت من" / "موت في"
    re.compile(r'[اأ]?م[وؤ]*ت\s+(في|فى|من|على|فيك|فيكي|فيها|فيه|عليك|عليكي)', re.IGNORECASE),
    # "هموت من الضحك" / "هموت من الحب"
    re.compile(r'هم[وؤ]*ت\s+(من|في|فى|على)', re.IGNORECASE),
    # "ميت من الضحك" / "ميتة من الخجل"
    re.compile(r'مي[تة]+\s+(من|في|فى|على)', re.IGNORECASE),
    # "يقتلني" = it kills me (figurative)
    re.compile(r'(بي|ي|ت)قتل(ني|نا|ك|ها|ه)', re.IGNORECASE),
    # "بحبك موت" = I love you to death
    re.compile(r'(بحبك?|بعشقك?|حبيبي|حبيبتي)\s*(موت|اموت|بموت)', re.IGNORECASE),
    re.compile(r'(موت|أموت|بموت)\s*(حب|عشق|ضحك|خجل|فرح|غيرة|جوع|برد|حر)', re.IGNORECASE),
    # "الشوق قاتلني" = longing is killing me
    re.compile(r'(الشوق|الحب|العشق|الضحك|الجوع|البرد|الحر)\s*(قاتل|بيقتل|يقتل)', re.IGNORECASE),
    # "ده يجنن" / "بتجنن" = this drives me crazy (positive)
    re.compile(r'(بتجنن|يجنن|مجنون\s*في|مجنونة\s*في)', re.IGNORECASE),

    # ---- English figurative / slang ----
    re.compile(r'(?:i\'?m\s+)?dying\s+(?:to|for|of|from|over|about)\b', re.IGNORECASE),
    re.compile(r'\b(?:kills?|killing|killed)\s+(?:me|it|the\s+game|the\s+vibe)\b', re.IGNORECASE),
    re.compile(r'\bdead\s*(?:ass)?\b', re.IGNORECASE),
    re.compile(r'\bi\'?m\s+dead\b', re.IGNORECASE),
    re.compile(r'\bdying\s+(?:laughing|of\s+laughter|rn|right\s+now)\b', re.IGNORECASE),
    re.compile(r'\bliterally\s+(?:dying|dead|killed\s+me)\b', re.IGNORECASE),
    re.compile(r'\bto\s+die\s+for\b', re.IGNORECASE),
    re.compile(r'\bkill(?:s|ed|ing)?\s+(?:the\s+)?(?:game|vibe|it|beat|performance|look)\b', re.IGNORECASE),
    re.compile(r'\bdrop\s*dead\s+gorgeous\b', re.IGNORECASE),
    re.compile(r'\bslaying\b', re.IGNORECASE),
    re.compile(r'\bkiller\s+(?:outfit|look|smile|move|song|beat|recipe|deal)\b', re.IGNORECASE),
    re.compile(r'\blove\s+(?:you|him|her|them|it)\s+to\s+(?:death|pieces|bits)\b', re.IGNORECASE),
    re.compile(r'\bcrazy\s+(?:about|for|over)\b', re.IGNORECASE),
    re.compile(r'\bhead\s+over\s+heels\b', re.IGNORECASE),
    re.compile(r'\bover\s+the\s+moon\b', re.IGNORECASE),
    re.compile(r'\bdying\s+(?:inside\s+)?(?:from|of)\s+(?:embarrassment|cringe|laughter|boredom|curiosity|excitement|anticipation|hunger|thirst)\b', re.IGNORECASE),
    re.compile(r'\bkill(?:ing)?\s+(?:time|two\s+birds)\b', re.IGNORECASE),
    re.compile(r'\bwould\s+(?:kill|die)\s+for\s+(?:a|some|that|this)\b', re.IGNORECASE),
    re.compile(r'\b(?:scared|bored|embarrassed|starving|freezing)\s+to\s+death\b', re.IGNORECASE),
    re.compile(r'\b(?:tickled|worried|worked)\s+to\s+death\b', re.IGNORECASE),

    # ---- French figurative ----
    re.compile(r'\bmourir\s+de\s+(?:rire|faim|soif|froid|chaud|honte|jalousie|ennui|envie|peur|amour)\b', re.IGNORECASE),
    re.compile(r"\bje\s+(?:meurs?|crève)\s+(?:de|d')\s*(?:rire|faim|soif|froid|chaud|honte|envie|amour)", re.IGNORECASE),
    re.compile(r'\bc\'?est\s+(?:à\s+)?mourir\s+de\s+rire\b', re.IGNORECASE),
    re.compile(r'\btu\s+me\s+(?:tues?|achèves?)\b', re.IGNORECASE),
    re.compile(r'\bje\s+(?:suis\s+)?(?:mort(?:e)?|crevé(?:e)?)\s+de\s+rire\b', re.IGNORECASE),
    re.compile(r'\bje\s+(?:t\'?aime|l\'?adore)\s+à\s+(?:mourir|en\s+mourir)\b', re.IGNORECASE),
    re.compile(r'\bfou(?:lle)?\s+(?:de|d\')\b', re.IGNORECASE),

    # ---- Spanish figurative ----
    re.compile(r'\bme\s+muero\s+(?:de|por)\s+(?:risa|hambre|sed|frío|calor|vergüenza|amor|ganas|sueño|celos|miedo|emoción)\b', re.IGNORECASE),
    re.compile(r'\bme\s+(?:mata(?:s)?|encanta|fascina)\b', re.IGNORECASE),
    re.compile(r'\bestoy\s+(?:muerto|muerta)\s+de\s+(?:risa|hambre|sueño|cansancio|frío|calor)\b', re.IGNORECASE),
    re.compile(r'\bme\s+muero\s+por\s+(?:ti|él|ella|eso|esto|comer|ver|ir|salir|conocer)\b', re.IGNORECASE),
    re.compile(r'(?:^|\s)loc[oa]\s+por\b', re.IGNORECASE),

    # ---- German figurative ----
    re.compile(r'\bich\s+(?:sterbe|krepiere)\s+vor\s+(?:lachen|hunger|durst|kälte|hitze|langeweile|neugier|angst|scham|liebe|sehnsucht)\b', re.IGNORECASE),
    re.compile(r'\btotlachen\b', re.IGNORECASE),
    re.compile(r'\bzum\s+(?:tot|sterben|schreien)\b', re.IGNORECASE),
    re.compile(r'\bich\s+(?:bin|war)\s+(?:gestorben|tot)\s+vor\s+lachen\b', re.IGNORECASE),
    re.compile(r'\bverrückt\s+nach\b', re.IGNORECASE),

    # ---- Italian figurative ----
    re.compile(r'(?:morire|muoio|moriamo)\s+(?:dal|di|dalle|da)\s+(?:ridere|risate|fame|sete|freddo|caldo|vergogna|sonno|noia|invidia|gelosia|amore)', re.IGNORECASE),
    re.compile(r'\bsto\s+morendo\s+(?:dal|di|dalle|da)\s+(?:ridere|fame|sete|freddo|caldo|sonno|noia)\b', re.IGNORECASE),
    re.compile(r'\bmi\s+(?:uccide|ammazza)\b', re.IGNORECASE),
    re.compile(r'(?:^|\s)pazz[oa]\s+(?:di|per)\b', re.IGNORECASE),

    # ---- Portuguese figurative ----
    re.compile(r'\b(?:tô|estou|to)\s+morrendo\s+de\s+(?:rir|fome|sede|frio|calor|vergonha|sono|tédio|medo|vontade|saudade|amor|ciúmes?)\b', re.IGNORECASE),
    re.compile(r'\bme\s+mata\b', re.IGNORECASE),
    re.compile(r'(?:^|\s)louc[oa]\s+por', re.IGNORECASE),

    # ---- Turkish figurative ----
    re.compile(r'\b(?:gülmekten|açlıktan|soğuktan|sıcaktan|sıkıntıdan|utançtan|uykusuzluktan|meraktan|korkudan|heyecandan|özlemden)\s+(?:öl|ölü|ölüyorum|ölüyoruz|öleceğim|ölecek)\b', re.IGNORECASE),
    re.compile(r'\böl[üu]yorum\s+(?:gülmekten|açlıktan|sıkıntıdan)\b', re.IGNORECASE),
    re.compile(r'\b(?:seni|onu|bunu)\s+(?:çok|aşırı|delicesine)\s+(?:seviyorum|istiyorum)\b', re.IGNORECASE),
    re.compile(r'\bdeli\s+(?:gibi|oldum|oluyor)\b', re.IGNORECASE),

    # ---- Japanese figurative ----
    re.compile(r'(?:笑い|恥ずかし|お腹|寒|暑|退屈|恋|好き|嬉し)(?:すぎ|で)(?:て)?(?:死|し)(?:ぬ|にそう|んだ|んでる)', re.IGNORECASE),
    re.compile(r'ウケ(?:る|死)', re.IGNORECASE),
    re.compile(r'(?:可愛|かわい|美味|うま|旨|最高|ヤバ|やば)(?:すぎ|くて)(?:て)?(?:死|し)(?:ぬ|にそう|んだ)', re.IGNORECASE),

    # ---- Chinese figurative ----
    re.compile(r'(?:笑|饿|冷|热|无聊|尴尬|想|爱|喜欢|紧张|兴奋|好吃|好看|可爱)(?:死|得要死|到死|得要命)', re.IGNORECASE),
    re.compile(r'我(?:太|好)?(?:爱|喜欢|迷)(?:死)?(?:了|你|他|她|它)', re.IGNORECASE),

    # ---- Korean figurative (bonus for robustness) ----
    re.compile(r'(?:웃겨|배고파|추워|더워|졸려|지루해|부끄러워|긴장돼)\s*(?:죽겠|죽을 것 같)', re.IGNORECASE),

    # ---- Urdu figurative ----
    re.compile(r'(?:ہنسی|بھوک|پیاس|سردی|گرمی|شرم|محبت|خوشی)\s*(?:سے|کی)\s*(?:مر|مار)', re.IGNORECASE),

    # ---- Hindi figurative ----
    re.compile(r'(?:हंसी|भूख|प्यास|ठंड|गर्मी|शर्म|प्यार|खुशी)\s*(?:से|की)\s*(?:मर|मार)', re.IGNORECASE),
    re.compile(r'(?:मर\s+रहा|मर\s+रही)\s+(?:हूं|हूँ)\s+(?:हंसी|भूख|प्यास|ठंड|गर्मी|शर्म)\s*(?:से|में)', re.IGNORECASE),

    # ---- Russian figurative ----
    re.compile(r'(?:умира|помира|сдох)\w*\s+(?:со\s+смеху|от\s+(?:смеха|голода|холода|жары|скуки|стыда|любви|нетерпения|страха|зависти|любопытства))', re.IGNORECASE),
    re.compile(r'(?:убивает|убила|убил)\s+(?:меня|наповал)', re.IGNORECASE),
    re.compile(r'(?:без\s+ума|схожу\s+с\s+ума|сума\s+сходить)\s+(?:от|по)', re.IGNORECASE),

    # ---- Vietnamese figurative ----
    re.compile(r'(?:chết|chết mất)\s+(?:cười|đói|khát|lạnh|nóng|buồn|ngủ|mệt|nhớ|yêu|thương|vui)', re.IGNORECASE),

    # ---- Swahili figurative ----
    re.compile(r'\bninakufa\s+(?:kwa\s+)?(?:kicheko|njaa|kiu|baridi|joto|aibu|upendo|furaha)\b', re.IGNORECASE),

    # ---- Thai figurative ----
    re.compile(r'(?:ตาย|ตายแล้ว)(?:ขำ|หัวเราะ|หิว|เหนื่อย|ร้อน|หนาว|อาย|รัก|ง่วง)', re.IGNORECASE),

    # ---- Polish figurative ----
    re.compile(r'\bumieram\s+(?:ze|z)\s+(?:śmiechu|głodu|pragnienia|zimna|gorąca|nudy|wstydu|ciekawości|strachu|miłości|zazdrości)\b', re.IGNORECASE),

    # ---- Dutch figurative ----
    re.compile(r'\bik\s+ga\s+dood\s+(?:van|van\s+het)\s+(?:lachen|honger|dorst|kou|warmte|verveling|schaamte|liefde|nieuwsgierigheid)\b', re.IGNORECASE),

    # ---- Greek figurative ----
    re.compile(r'(?:πεθαίνω|σκοτώνομαι)\s+(?:στα\s+γέλια|από\s+(?:τα\s+γέλια|πείνα|δίψα|κρύο|ζέστη|ντροπή|αγάπη|περιέργεια))', re.IGNORECASE),
    re.compile(r'\bτρελός\s+(?:για|με)\b', re.IGNORECASE),
]


def is_figurative_expression(text: str) -> bool:
    """Check if text contains figurative/idiomatic use of death/harm words.
    
    Returns True if the text matches known figurative patterns, meaning
    it should NOT be classified as crisis or mental health.
    """
    for pattern in _FIGURATIVE_PATTERNS:
        if pattern.search(text):
            return True
    return False


class Intent(BaseModel):
    type: Literal[
        "greeting",
        "goodbye",
        "gratitude",
        "out_of_scope",
        "asking_mental_health_question",
        "crisis",
    ] = Field(
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


class IntentClassifier:
    """Intent classifier using embeddings and LLM fallback."""

    def __init__(self, llm_fallback: Callable[[str], Intent] | None = None):
        self.llm_fallback = llm_fallback or self._llm_fallback

        # Initialize DSPy LM and clients (Lightning AI preferred, fallback to Groq)
        lightning_api_key = config.LIGHTNING_API_KEY
        if lightning_api_key:
            self.model_name = config.LIGHTNING_CLASSIFIER_MODEL
            dspy_model = self.model_name if self.model_name.startswith("openai/") else f"openai/{self.model_name}"
            self.lm = dspy.LM(dspy_model, api_key=lightning_api_key, api_base="https://lightning.ai/api/v1")
            self.fallback_module = IntentClassifierModule()
            
            # Setup OpenAI client for compatibility
            from openai import OpenAI
            self.groq_client = OpenAI(api_key=lightning_api_key, base_url="https://lightning.ai/api/v1")
        elif config.GROQ_API_KEY:
            self.model_name = config.GROQ_CLASSIFIER_MODEL
            dspy_model = self.model_name if self.model_name.startswith("groq/") else f"groq/{self.model_name}"
            self.lm = dspy.LM(dspy_model, api_key=config.GROQ_API_KEY)
            self.fallback_module = IntentClassifierModule()
            
            # Setup Groq client
            from groq import Groq
            self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        else:
            self.model_name = "lightning-ai/gpt-oss-20b"
            self.lm = None
            self.fallback_module = None
            self.groq_client = None

        self.embedding_examples = {
            "greeting": [
                "hello",
                "hi there",
                "hey",
                "greetings",
                "good morning",
                "good afternoon",
                "good evening",
                "hiya",
                "hello, how are you",
                "how are you",
                "how's it going",
                "how are you doing",
                "how are yoy",
                "nice to meet you",
                "my name is john",
                "i am john",
                "call me john",
                "what is my name",
                "who am i",
                "my name is",
                "tell me my name",
                "who are you",
                "what is your name",
                "what can you do",
                "how can you help me",
                "مرحبا",
                "أهلاً",
                "اهلا",
                "السلام عليكم",
                "سلام عليكم",
                "صباح الخير",
                "كيف حالك",
                "اسمي احمد",
                "انا احمد",
                "ما هو اسمي",
                "من انت",
                "ما اسمك",
                "ہیلو",
                "السلام علیکم",
                "آپ کیسے ہیں",
                "میرا نام علی ہے",
                "میرا نام کیا ہے",
                "hola",
                "buenos días",
                "buenas tardes",
                "buenas noches",
                "bonjour",
                "salut",
                "bonsoir",
                "ciao",
                "buongiorno",
                "你好",
            ],
            "goodbye": [
                "goodbye",
                "bye",
                "see you",
                "take care",
                "farewell",
                "bye bye",
                "مع السلامة",
                "إلى اللقاء",
                "وداعا",
                "خدا حافظ",
                "tschüss",
                "auf wiedersehen",
                "au revoir",
                "adiós",
                "arrivederci",
                "再见",
            ],
            "gratitude": [
                "thanks",
                "thank you",
                "thx",
                "appreciate it",
                "thank you so much",
                "شكرا",
                "شكراً",
                "تسلم",
                "جزاك الله خيراً",
                "شکریہ",
                "بہت شکریہ",
                "danke",
                "merci",
                "gracias",
                "grazie",
                "asante",
                "谢谢",
            ],
            "out_of_scope": [
                # General off-topic (English)
                "what is the weather like",
                "tell me a joke",
                "what is the latest news",
                "what sports scores are",
                "what is the capital of France",
                "how to cook pasta",
                "write me a python script",
                "solve this math problem",
                "who won the world cup",
                "what time is it",
                "recommend a movie",
                "what is machine learning",
                "tell me about history",
                "how does a car engine work",
                "what is the stock price of apple",
                "translate this to French",
                "help me with my homework",
                "write an essay about climate change",
                "how to fix a leaky faucet",
                "best restaurants near me",
                # Figurative / slang (English) — NOT crisis
                "I'm dying for a pizza",
                "this song kills me",
                "I'm dead, that was so funny",
                "she's killing it on stage",
                "dying of laughter right now",
                "I would kill for some coffee",
                "that joke killed me",
                "I'm dying to see that movie",
                "drop dead gorgeous",
                "love you to death",
                "bored to death",
                "scared to death of spiders",
                "starving to death over here",
                "he's crazy about her",
                "I'm over the moon",
                "killing time at the airport",
                # Figurative / slang (Arabic) — NOT crisis
                "أنا بموت في مازن",
                "بموت فيك يا حبيبي",
                "بموت من الضحك",
                "بمووت في الأكل ده",
                "أنا بموت على الشوكولاتة",
                "هموت من الضحك",
                "ميتة من الخجل",
                "الأغنية دي بتقتلني",
                "مجنون فيها",
                "بحبك موت",
                "الشوق قاتلني",
                "أنا بموت في الأكل الحار",
                "بموت عليكي",
                "ده يجنن من جماله",
                # Figurative / slang (French) — NOT crisis
                "je meurs de rire",
                "c'est à mourir de rire",
                "tu me tues avec tes blagues",
                "je suis mort de rire",
                "je meurs de faim",
                "je t'aime à mourir",
                "fou de toi",
                # Figurative / slang (Spanish) — NOT crisis
                "me muero de risa",
                "me muero de hambre",
                "me muero por ti",
                "me mata tu sonrisa",
                "estoy muerto de risa",
                "loca por él",
                # Figurative / slang (German) — NOT crisis
                "ich sterbe vor lachen",
                "ich sterbe vor hunger",
                "das ist zum totlachen",
                "verrückt nach dir",
                # Figurative / slang (Italian) — NOT crisis
                "muoio dal ridere",
                "sto morendo di fame",
                "mi uccide questa canzone",
                "pazza di te",
                # Figurative / slang (Portuguese) — NOT crisis
                "tô morrendo de rir",
                "morrendo de fome",
                "louca por você",
                # Figurative / slang (Turkish) — NOT crisis
                "gülmekten öldüm",
                "açlıktan ölüyorum",
                "seni çok seviyorum",
                # Figurative / slang (Russian) — NOT crisis
                "умираю со смеху",
                "убила меня этим",
                "без ума от тебя",
                # Figurative / slang (Chinese) — NOT crisis
                "笑死我了",
                "饿死了",
                "好看死了",
                "我太爱了",
                # Figurative / slang (Japanese) — NOT crisis
                "笑いすぎて死にそう",
                "お腹すいて死にそう",
                "可愛すぎて死ぬ",
                # General off-topic (Arabic)
                "ما هو الطقس اليوم",
                "اخبرني بنكتة",
                "ما هي آخر الأخبار",
                "ما هي عاصمة فرنسا",
                "كيف أطبخ المعكرونة",
                "من فاز بكأس العالم",
                # General off-topic (Urdu)
                "آج موسم کیسا ہے",
                "مجھے لطیفہ سنائیں",
                "تازہ ترین خبریں کیا ہیں",
                # General off-topic (French)
                "quel temps fait-il",
                "raconte-moi une blague",
                # General off-topic (Spanish)
                "cómo está el clima",
                "cuéntame un chiste",
                # Romance / relationship (not mental health)
                "I love my girlfriend so much",
                "أنا بحب خطيبتي أوي",
                "je suis amoureux d'elle",
                "estoy enamorado de ella",
            ],
            "asking_mental_health_question": [
                "I feel anxious and need help",
                "I am depressed",
                "can you help me with my stress",
                "I need therapy advice",
                "I've been feeling really down lately",
                "how to deal with panic attacks",
                "what are signs of burnout",
                "I can't sleep because of worry",
                "أشعر بالقلق وأحتاج إلى مساعدة",
                "أنا مكتئب وأحتاج للتحدث مع شخص ما",
                "ساعدني في التغلب على التوتر",
                "میں فکر مند محسوس کرتا ہوں اور مجھے مدد کی ضرورت ہے",
                "میں اداس اور افسردہ ہوں",
                "کیا آپ میرے تناؤ میں مدد کر سکتے ہیں",
                "je me sens anxieux",
                "je suis déprimé",
                "me siento ansioso",
                "estoy deprimido",
            ],
            "crisis": [
                "I want to kill myself",
                "I want to end my life",
                "I am thinking of suicide",
                "I want to cut myself",
                "how to end my life",
                "harm myself",
                "أريد الانتحار",
                "أريد إنهاء حياتي",
                "أفكر في الانتحار",
                "أريد إيذاء نفسي",
                "میں خودکشی کرنا چاہتا ہوں",
                "میں اپنی زندگی ختم کرنا چاہتا ہوں",
                "میں خود کو نقصان پہنچانا چاہتا ہوں",
                "je veux me tuer",
                "je veux en glorie avec ma vie",
                "quiero suicidarme",
                "quiero terminar con mi vida",
            ],
        }

        self.embedding_threshold = 0.85
        self.embedding_model = (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        hf_token = config.HF_TOKEN
        self.embedding_client = InferenceClient(
            provider="hf-inference",
            api_key=hf_token,
        )

        try:
            from fastembed import TextEmbedding

            print("--> [Intent Classifier] Loading fastembed text-embedding model...")
            self.local_model = TextEmbedding(self.embedding_model)
            self.use_local = True
        except Exception as e:
            print(
                f"--> [Intent Classifier] Local fastembed not used: {e}. Using HF API Client."
            )
            self.use_local = False

        # Precompute normalized embeddings for example sentences
        if self.use_local:
            self.embedding_examples_embeddings = {
                intent_type: self._normalize_embeddings(
                    np.asarray(
                        list(self.local_model.embed(examples)),
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
        # =================================================================
        # GUARDRAIL LAYER 0: Figurative / Idiomatic Expression Pre-Check
        # =================================================================
        # This runs BEFORE any ML model. It catches slang like:
        #   - "أنا بموت في مازن" (I'm crazy about Mazen)
        #   - "I'm dying for pizza"
        #   - "je meurs de rire" (I'm dying of laughter)
        #   - "me muero de hambre" (I'm dying of hunger)
        # These are NOT crisis, NOT mental health — they are out_of_scope.
        # =================================================================
        if is_figurative_expression(text):
            print(
                f"---> [Intent Classifier] GUARDRAIL: Figurative/idiomatic expression detected. Classifying as 'out_of_scope'."
            )
            return Intent(
                type="out_of_scope",
                confidence=0.99,
                classifier="embedding",
            )

        # Run embedding classifier for all languages since we now use a multilingual embedding model
        embedding_intent = self._embedding_classifier(text)
        
        resolved_intent = None
        if embedding_intent is not None:
            # If the predicted intent is crisis but confidence is below 85%, route to LLM fallback
            if embedding_intent.type == "crisis" and embedding_intent.confidence < 0.85:
                print(
                    f"---> [Intent Classifier] Embedding classified as crisis but confidence {embedding_intent.confidence:.5f} is below 85%. Routing to LLM fallback."
                )
                resolved_intent = self.llm_fallback(text)
            else:
                print(
                    f"---> [Intent Classifier] Classified intent as '{embedding_intent.type}' with confidence {embedding_intent.confidence:.5f} using embedding classifier."
                )
                resolved_intent = embedding_intent
        else:
            # LLM fallback
            resolved_intent = self.llm_fallback(text)

        # =================================================================
        # GUARDRAIL LAYER 3: Post-Classification Crisis Safety Override
        # =================================================================
        # If the resolved intent is "crisis", but detect_crisis(text) is False,
        # override it to "asking_mental_health_question".
        if resolved_intent.type == "crisis":
            from .rag import detect_crisis
            if not detect_crisis(text):
                print(
                    f"---> [Intent Classifier] Overriding intent 'crisis' to 'asking_mental_health_question' because detect_crisis returned False."
                )
                resolved_intent.type = "asking_mental_health_question"
                resolved_intent.confidence = 0.90
                
        return resolved_intent

    def _embedding_classifier(self, text: str) -> Intent | None:
        text_embedding = self._get_embedding(text)
        best_intent = None
        best_score = 0.0

        for (
            intent_type,
            example_embeddings,
        ) in self.embedding_examples_embeddings.items():
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
                    type=res["type"], confidence=res["confidence"], classifier="llm"
                )
            except Exception as e:
                try:
                    print(
                        f"--> [Intent Classifier Error] DSPy API fallback failed: {e}"
                    )
                except Exception:
                    pass

        # Return default if Groq fails or is not available
        return Intent(type="greeting", confidence=0.5, classifier="llm")

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.use_local:
            embedding = list(self.local_model.embed([text]))[0]
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
        use_rag = intent == "asking_mental_health_question"

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
            template = (
                "Goodbye! Take care of yourself. I am always here if you need support."
            )
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

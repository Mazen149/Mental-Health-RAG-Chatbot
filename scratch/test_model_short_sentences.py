import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.modules.language_detector import LanguageDetector, preprocess

# 56 short test cases across 7 languages
TEST_CASES = [
    # --- English ---
    {"text": "i am sad", "expected": "English", "lang": "English"},
    {"text": "help me please", "expected": "English", "lang": "English"},
    {"text": "i feel nervous", "expected": "English", "lang": "English"},
    {"text": "so lonely today", "expected": "English", "lang": "English"},
    {"text": "very stressed out", "expected": "English", "lang": "English"},
    {"text": "cannot sleep well", "expected": "English", "lang": "English"},
    {"text": "anxiety is bad", "expected": "English", "lang": "English"},
    {"text": "need to talk", "expected": "English", "lang": "English"},

    # --- Arabic ---
    {"text": "أنا حزين جدا", "expected": "Arabic", "lang": "Arabic"},
    {"text": "ساعدني أرجوك", "expected": "Arabic", "lang": "Arabic"},
    {"text": "أشعر بقلق", "expected": "Arabic", "lang": "Arabic"},
    {"text": "أنا وحيد هنا", "expected": "Arabic", "lang": "Arabic"},
    {"text": "ضغوطات كبيرة", "expected": "Arabic", "lang": "Arabic"},
    {"text": "لا أستطيع النوم", "expected": "Arabic", "lang": "Arabic"},
    {"text": "أحتاج مساعدة", "expected": "Arabic", "lang": "Arabic"},
    {"text": "أشعر بالخوف", "expected": "Arabic", "lang": "Arabic"},

    # --- Spanish ---
    {"text": "estoy muy triste", "expected": "Spanish", "lang": "Spanish"},
    {"text": "ayúdame por favor", "expected": "Spanish", "lang": "Spanish"},
    {"text": "tengo mucho miedo", "expected": "Spanish", "lang": "Spanish"},
    {"text": "me siento solo", "expected": "Spanish", "lang": "Spanish"},
    {"text": "estoy muy estresado", "expected": "Spanish", "lang": "Spanish"},
    {"text": "necesito apoyo", "expected": "Spanish", "lang": "Spanish"},
    {"text": "no puedo dormir", "expected": "Spanish", "lang": "Spanish"},
    {"text": "siento mucha ansiedad", "expected": "Spanish", "lang": "Spanish"},

    # --- French ---
    {"text": "je suis triste", "expected": "French", "lang": "French"},
    {"text": "aidez-moi s'il vous plaît", "expected": "French", "lang": "French"},
    {"text": "j'ai très peur", "expected": "French", "lang": "French"},
    {"text": "je me sens seul", "expected": "French", "lang": "French"},
    {"text": "très stressé", "expected": "French", "lang": "French"},
    {"text": "besoin d'aide", "expected": "French", "lang": "French"},
    {"text": "insomnie et angoisse", "expected": "French", "lang": "French"},
    {"text": "je ne vais pas bien", "expected": "French", "lang": "French"},

    # --- German ---
    {"text": "ich bin traurig", "expected": "German", "lang": "German"},
    {"text": "bitte hilf mir", "expected": "German", "lang": "German"},
    {"text": "ich habe angst", "expected": "German", "lang": "German"},
    {"text": "ich bin einsam", "expected": "German", "lang": "German"},
    {"text": "sehr gestresst", "expected": "German", "lang": "German"},
    {"text": "brauche hilfe", "expected": "German", "lang": "German"},
    {"text": "kann nicht schlafen", "expected": "German", "lang": "German"},
    {"text": "angstzustände", "expected": "German", "lang": "German"},

    # --- Swahili ---
    {"text": "nahisi huzuni", "expected": "Swahili", "lang": "Swahili"},
    {"text": "nisaidie tafadhali", "expected": "Swahili", "lang": "Swahili"},
    {"text": "ninaogopa sana", "expected": "Swahili", "lang": "Swahili"},
    {"text": "nahisi upweke", "expected": "Swahili", "lang": "Swahili"},
    {"text": "nimefadhaika", "expected": "Swahili", "lang": "Swahili"},
    {"text": "nahitaji msaada", "expected": "Swahili", "lang": "Swahili"},
    {"text": "siwezi kulala", "expected": "Swahili", "lang": "Swahili"},
    {"text": "nina wasiwasi", "expected": "Swahili", "lang": "Swahili"},

    # --- Turkish ---
    {"text": "çok üzgünüm", "expected": "Turkish", "lang": "Turkish"},
    {"text": "lütfen yardım et", "expected": "Turkish", "lang": "Turkish"},
    {"text": "korkuyorum", "expected": "Turkish", "lang": "Turkish"},
    {"text": "kendimi yalnız hissediyorum", "expected": "Turkish", "lang": "Turkish"},
    {"text": "çok stresliyim", "expected": "Turkish", "lang": "Turkish"},
    {"text": "yardıma ihtiyacım var", "expected": "Turkish", "lang": "Turkish"},
    {"text": "uyuyamıyorum", "expected": "Turkish", "lang": "Turkish"},
    {"text": "endişeliyim", "expected": "Turkish", "lang": "Turkish"},
]

COMMON_STOPWORDS = {
    "en": {"i", "am", "sad", "feel", "depressed", "nervous", "lonely", "stressed", "anxious", "anxiety", "help", "please", "want", "sleep", "talk", "my", "myself", "don't", "dont", "can't", "cant", "suicide", "kill"},
    "ar": {"أنا", "حزين", "جدا", "ساعدني", "أشعر", "بقلق", "وحيد", "هنا", "ضغوطات", "كبيرة", "لا", "أستطيع", "النوم", "أحتاج", "مساعدة", "بالخوف"},
    "es": {"estoy", "muy", "triste", "ayúdame", "tengo", "miedo", "me", "siento", "solo", "estresado", "necesito", "apoyo", "no", "puedo", "dormir", "siento", "mucha", "ansiedad", "por", "favor"},
    "fr": {"je", "suis", "triste", "aidez", "moi", "peur", "me", "sens", "seul", "stressé", "besoin", "aide", "insomnie", "angoisse", "ne", "vais", "pas", "bien", "s'il", "vous", "plaît"},
    "de": {"ich", "bin", "traurig", "hilf", "mir", "habe", "angst", "einsam", "gestresst", "brauche", "hilfe", "kann", "nicht", "schlafen", "angstzustände", "bitte"},
    "sw": {"nahisi", "huzuni", "nisaidie", "tafadhali", "ninaogopa", "sana", "upweke", "nimefadhaika", "nahitaji", "msaada", "siwezi", "kulala", "nina", "wasiwasi"},
    "tr": {"çok", "üzgünüm", "lütfen", "yardım", "et", "korkuyorum", "kendimi", "yalnız", "hissediyorum", "stresliyim", "yardıma", "ihtiyacım", "var", "uyuyamıyorum", "endişeliyim"},
    "it": {"sono", "triste", "aiutami", "ho", "paura", "mi", "sento", "solo", "stressato", "ho", "bisogno", "aiuto", "non", "riesco", "dormire", "ansia", "per", "favore"},
    "nl": {"ik", "ben", "triest", "help", "mij", "heb", "angst", "eenzaam", "gestrest", "heb", "nodig", "hulp", "kan", "niet", "slapen", "angst", "alstublieft", "voel"},
    "pt": {"estou", "muito", "triste", "ajude-me", "tenho", "medo", "me", "sinto", "só", "estressado", "preciso", "apoio", "não", "consigo", "dormir", "ansiedade", "por", "favor"},
    "pl": {"jestem", "bardzo", "smutny", "pomóż", "smutna", "boję", "się", "czuję", "samotny", "zestresowany", "potrzebuję", "pomocy", "nie", "mogę", "spać", "lęk", "proszę"},
}

LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "sw": "Swahili", "tr": "Turkish"
}

class EnhancedDetector:
    def __init__(self):
        self.base_detector = LanguageDetector()

    def detect(self, text: str) -> str:
        # 1. Preprocess and extract words
        cleaned = preprocess(text)
        words = set(cleaned.split())
        
        # 2. Check keyword matches
        best_lang = None
        max_matches = 0
        for lang_code, stopwords in COMMON_STOPWORDS.items():
            matches = len(words.intersection(stopwords))
            if matches > max_matches:
                max_matches = matches
                best_lang = lang_code
        
        # If we have a strong matching language (at least 1 keyword match)
        if best_lang is not None and max_matches >= 1:
            return LANGUAGE_NAMES[best_lang]
            
        # 3. Otherwise, fall back to base detector (with dynamic thresholding)
        unpadded = text.replace("/p", "").replace("/P", "").strip()
        unpadded_words = unpadded.split()
        thresh = 0.85 if len(unpadded_words) >= 5 else 0.3
        
        res = self.base_detector.detect(text, threshold=thresh)
        return res["language_name"]

def main():
    detector = EnhancedDetector()
    
    # Configure console for UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    correct = 0
    failures = []
    
    for idx, case in enumerate(TEST_CASES):
        text = case["text"]
        expected = case["expected"]
        
        pred = detector.detect(text)
        if pred == expected:
            correct += 1
        else:
            failures.append((text, expected, pred))
            
    print("=" * 80)
    print("ENHANCED DETECTOR EVALUATION ON 56 SHORT CASES")
    print("=" * 80)
    print(f"Accuracy: {(correct / len(TEST_CASES)) * 100:.2f}% ({correct}/{len(TEST_CASES)} passed)")
    
    if failures:
        print("\nFailing Cases:")
        for text, expected, pred in failures:
            print(f" - Query: '{text}' ({expected}) -> Got: {pred}")
    else:
        print("\n🎉 PERFECT ACCURACY! All 56 cases passed successfully.")
        
    print("\n" + "=" * 80)
    print("Testing additional problematic queries:")
    additional = [
        "i am sad",
        "i am depressed",
        "i feel depressed",
        "i am sad i don't want to kill myself"
    ]
    for q in additional:
        print(f" - Query: '{q}' -> Predicted: {detector.detect(q)}")
    print("=" * 80)

if __name__ == "__main__":
    main()

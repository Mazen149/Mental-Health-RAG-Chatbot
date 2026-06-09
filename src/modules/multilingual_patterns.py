"""
================================================================================
SERENE AI — MULTILINGUAL PATTERNS & RESPONSE TEMPLATES
================================================================================
Contains localized regex patterns, conversational templates, crisis hotline
responses, and polite out-of-scope redirection text for all 20 languages.
================================================================================
"""

import re

# 20 supported language names matching language_detector.py keys
LANGUAGES = [
    "Arabic", "Bulgarian", "German", "Greek", "English", "Spanish", "French", 
    "Hindi", "Italian", "Japanese", "Dutch", "Polish", "Portuguese", "Russian", 
    "Swahili", "Thai", "Turkish", "Urdu", "Vietnamese", "Chinese"
]
# Multilingual greeting patterns by language for exact language matching
GREETING_PATTERNS_BY_LANG = {
    "Arabic": [r"مرحبا", r"أهلاً", r"اهلا", r"السلام\s*عليكم", r"سلام\s*عليكم", r"سلام"],
    "Bulgarian": [r"здравей", r"здравейте", r"добър\s*ден"],
    "German": [r"hallo", r"guten\s*tag", r"hi", r"grüß\s*gott"],
    "Greek": [r"γεια", r"χαίρετε", r"καλημέρα"],
    "English": [r"hello", r"hi", r"hey", r"greetings", r"good\s*morning", r"good\s*afternoon", r"good\s*evening", r"hiya", r"welcome"],
    "Spanish": [r"hola", r"buenos\s*días", r"buenos\s*dias", r"buenas\s*tardes", r"buenas\s*noches"],
    "French": [r"bonjour", r"salut", r"bonsoir", r"coucou"],
    "Hindi": [r"नमस्ते", r"नमस्कार", r"हैलो"],
    "Italian": [r"ciao", r"salve", r"buongiorno"],
    "Japanese": [r"こんにちは", r"おはよう", r"こんばんは"],
    "Dutch": [r"hallo", r"hoi", r"goedemorgen", r"goedenavond"],
    "Polish": [r"cześć", r"czesc", r"dzień\s*dobry", r"dzien\s*dobry", r"witaj"],
    "Portuguese": [r"olá", r"ola", r"oi", r"bom\s*dia", r"boa\s*tarde"],
    "Russian": [r"привет", r"здравствуйте", r"доброе\s*утро"],
    "Swahili": [r"jambo", r"mambo", r"habari"],
    "Thai": [r"สวัสดี", r"หวัดดี"],
    "Turkish": [r"merhaba", r"selam", r"günaydın", r"gunaydin"],
    "Urdu": [r"ہیلو", r"السلام\s*علیکم", r"آداب"],
    "Vietnamese": [r"xin\s*chào", r"xin\s*chao", r"chào\s*bạn", r"chao\s*ban"],
    "Chinese": [r"你好", r"您好", r"早上好", r"下午好", r"晚上好"]
}

# Multilingual goodbye patterns by language for exact language matching
GOODBYE_PATTERNS_BY_LANG = {
    "Arabic": [r"مع\s*السلامة", r"إلى\s*اللقاء", r"الى\s*اللقاء", r"وداعا", r"وداعاً"],
    "Bulgarian": [r"довиждане", r"чао"],
    "German": [r"tschüss", r"tschuss", r"auf\s*wiedersehen", r"ciao"],
    "Greek": [r"αντίο", r"γεια\s*σας", r"τα\s*λέμε"],
    "English": [r"goodbye", r"bye", r"see\s*you", r"take\s*care", r"farewell"],
    "Spanish": [r"adiós", r"adios", r"hasta\s*luego", r"nos\s*vemos"],
    "French": [r"au\s*revoir", r"salut", r"à\s*bientôt", r"a\s*bientot"],
    "Hindi": [r"अलविदा", r"फिर\s*मिलेंगे"],
    "Italian": [r"arrivederci", r"ciao", r"a\s*presto"],
    "Japanese": [r"さようなら", r"じゃあね", r"バイバイ"],
    "Dutch": [r"doeg", r"tot\s*ziens", r"dag"],
    "Polish": [r"do\s*widzenia", r"pa", r"na\s*razie"],
    "Portuguese": [r"adeus", r"tchau", r"até\s*logo", r"ate\s*logo"],
    "Russian": [r"пока", r"до\s*свидания", r"до\s*встречи"],
    "Swahili": [r"kwa\s*heri", r"tutaonana"],
    "Thai": [r"ลาก่อน"],
    "Turkish": [r"hoşça\s*kal", r"hosca\s*kal", r"güle\s*güle", r"gule\s*gule", r"görüşürüz", r"gorusuruz"],
    "Urdu": [r"خدا\s*حافظ", r"الوداع"],
    "Vietnamese": [r"tạm\s*biệt", r"tam\s*biet", r"hẹn\s*gặp\s*lại", r"hen\s*gap\s*lai"],
    "Chinese": [r"再见", r"拜拜"]
}

# Multilingual gratitude patterns by language for exact language matching
GRATITUDE_PATTERNS_BY_LANG = {
    "Arabic": [r"شكرا", r"شكراً", r"شكر", r"جزاك\s*الله", r"تسلم"],
    "Bulgarian": [r"благodarя", r"мерси", r"благodarности"],
    "German": [r"danke", r"vielen\s*dank"],
    "Greek": [r"ευχαριστώ", r"ευχαριστω"],
    "English": [r"thanks", r"thank\s*you", r"thx", r"appreciate\s*it"],
    "Spanish": [r"gracias", r"muchas\s*gracias"],
    "French": [r"merci", r"merci\s*beaucoup"],
    "Hindi": [r"धन्यवाद", r"शुक्रिया"],
    "Italian": [r"grazie", r"grazie\s*mille"],
    "Japanese": [r"ありがとう", r"ありがとうございます", r"どうも", r"感謝します"],
    "Dutch": [r"dank\s*je", r"bedankt", r"dank\s*u"],
    "Polish": [r"dziękuję", r"dziekuje", r"dzięki", r"dzeki"],
    "Portuguese": [r"obrigado", r"obrigada"],
    "Russian": [r"спасибо", r"благодарю"],
    "Swahili": [r"asante", r"asante\s*sana"],
    "Thai": [r"ขอบคุณ", r"ขอบใจ"],
    "Turkish": [r"teşekkürler", r"tesekkurler", r"teşekkür\s*ederim", r"tesekkur\s*ederim", r"sağol", r"sagol"],
    "Urdu": [r"شکریہ", r"بہت\s*شکریہ"],
    "Vietnamese": [r"cảm\s*ơn", r"cam\s*on", r"cám\s*ơn", r"cam\s*on\s*ban"],
    "Chinese": [r"谢谢", r"谢谢你"]
}

# Flatten lists to maintain backward compatibility
GREETING_PATTERNS = []
for patterns in GREETING_PATTERNS_BY_LANG.values():
    GREETING_PATTERNS.extend(patterns)

GOODBYE_PATTERNS = []
for patterns in GOODBYE_PATTERNS_BY_LANG.values():
    GOODBYE_PATTERNS.extend(patterns)

GRATITUDE_PATTERNS = []
for patterns in GRATITUDE_PATTERNS_BY_LANG.values():
    GRATITUDE_PATTERNS.extend(patterns)

# Compile flat regex patterns
GREETING_REGEX = re.compile(r"^(" + "|".join(GREETING_PATTERNS) + r")(\s+.*)?$", re.IGNORECASE)
GOODBYE_REGEX = re.compile(r"^(" + "|".join(GOODBYE_PATTERNS) + r")(\s+.*)?$", re.IGNORECASE)
GRATITUDE_REGEX = re.compile(r"^(" + "|".join(GRATITUDE_PATTERNS) + r")(\s+.*)?$", re.IGNORECASE)

# Compile language-specific regex patterns
GREETING_REGEX_BY_LANG = {
    lang: re.compile(r"^(" + "|".join(patterns) + r")(\s+.*)?$", re.IGNORECASE)
    for lang, patterns in GREETING_PATTERNS_BY_LANG.items()
}
GOODBYE_REGEX_BY_LANG = {
    lang: re.compile(r"^(" + "|".join(patterns) + r")(\s+.*)?$", re.IGNORECASE)
    for lang, patterns in GOODBYE_PATTERNS_BY_LANG.items()
}
GRATITUDE_REGEX_BY_LANG = {
    lang: re.compile(r"^(" + "|".join(patterns) + r")(\s+.*)?$", re.IGNORECASE)
    for lang, patterns in GRATITUDE_PATTERNS_BY_LANG.items()
}


# Direct friendly responses for each language
GREETING_RESPONSES = {
    "Arabic": "مرحباً! أنا هنا لدعمك ومساعدتك. كيف يمكنني مساعدتك اليوم؟",
    "Bulgarian": "Здравейте! Аз съм тук, за да ви подкрепя. Как мога да ви помогна днес?",
    "German": "Hallo! Ich bin hier, um Sie zu unterstützen. Wie kann ich Ihnen heute helfen?",
    "Greek": "Γεια σας! Είμαι εδώ για να σας υποστηρίξω. Πώς μπορώ να σας βοηθήσω σήμερα;",
    "English": "Hello! I am here to support you. How can I help you today?",
    "Spanish": "¡Hola! Estoy aquí para apoyarte. ¿Cómo puedo ayudarte hoy?",
    "French": "Bonjour ! Je suis là pour vous soutenir. Comment puis-je vous aider aujourd'hui ?",
    "Hindi": "नमस्ते! मैं आपकी सहायता के लिए यहाँ हूँ। आज मैं आपकी क्या मदद कर सकता हूँ?",
    "Italian": "Ciao! Sono qui per supportarti. Come posso aiutarti oggi?",
    "Japanese": "こんにちは！私はあなたをサポートするためにここにいます。今日はどのようなご相談でしょうか？",
    "Dutch": "Hallo! Ik ben hier om je te ondersteunen. Hoe kan ik je vandaag helpen?",
    "Polish": "Cześć! Jestem tutaj, aby Cię wesprzeć. W czym mogę Ci dzisiaj pomóc?",
    "Portuguese": "Olá! Estou aqui para apoiar você. Como posso ajudar hoje?",
    "Russian": "Привет! Я здесь, чтобы поддержать вас. Чем я могу помочь вам сегодня?",
    "Swahili": "Jambo! Niko hapa kukusaidia. Ninawezaje kukusaidia leo?",
    "Thai": "สวัสดีค่ะ/ครับ! ฉันอยู่ที่นี่เพื่อสนับสนุนคุณ วันนี้มีอะไรให้ฉันช่วยเหลือไหมคะ/ครับ?",
    "Turkish": "Merhaba! Size destek olmak için buradayım. Bugün size nasıl yardımcı olabilirim?",
    "Urdu": "ہیلو! میں آپ کی مدد کے لیے حاضر ہوں۔ آج میں آپ کی کیا مدد کر سکتا ہوں؟",
    "Vietnamese": "Xin chào! Tôi ở đây để hỗ trợ bạn. Tôi có thể giúp gì cho bạn hôm nay?",
    "Chinese": "你好！我在这里为您提供支持。今天有什么我可以帮您的吗？"
}

GOODBYE_RESPONSES = {
    "Arabic": "مع السلامة! أتمنى لك يوماً هادئاً ومليئاً بالسكينة. أنا هنا دائماً إذا احتجت للدعم.",
    "Bulgarian": "Довиждане! Грижете се за себе си. Винаги съм тук, ако имате нужда от подкрепа.",
    "German": "Auf Wiedersehen! Passen Sie auf sich auf. Ich bin jederzeit für Sie da.",
    "Greek": "Αντίο! Να προσέχετε τον εαυτό σας. Είμαι πάντα εδώ αν χρειαστείτε υποστήριξη.",
    "English": "Goodbye! Take care of yourself. I am always here if you need support.",
    "Spanish": "¡Adiós! Cuídate mucho. Estoy aquí si necesitas apoyo.",
    "French": "Au revoir ! Prenez soin de vous. Je suis là si vous avez besoin de soutien.",
    "Hindi": "नमस्ते! अपना ख्याल रखें। यदि आपको कभी भी सहायता की आवश्यकता हो, तो मैं हमेशा यहाँ हूँ।",
    "Italian": "Ciao! Prenditi cura di te. Sono qui se hai bisogno di supporto.",
    "Japanese": "さようなら！お体に気をつけてください。サポートが必要なときはいつでもここにいます。",
    "Dutch": "Tot ziens! Zorg goed voor jezelf. Ik ben er altijd als je ondersteuning nodig hebt.",
    "Polish": "Do widzenia! Dbaj o siebie. Jestem tutaj zawsze, gdy potrzebujesz wsparcia.",
    "Portuguese": "Adeus! Cuide-se bem. Estarei aqui sempre que precisar de apoio.",
    "Russian": "До свидания! Берегите себя. Я всегда здесь, если вам понадобится поддержка.",
    "Swahili": "Kwa heri! Jilinde. Niko hapa kila wakati ukihitaji usaidizi.",
    "Thai": "ลาก่อนค่ะ/ครับ! ดูแลตัวเองด้วยนะ ฉันพร้อมสนับสนุนคุณเสมอหากต้องการ",
    "Turkish": "Hoşça kalın! Kendinize iyi bakın. Desteğe ihtiyaç duyduğunuzda her zaman buradayım.",
    "Urdu": "خدا حافظ! اپنا خیال رکھیں۔ جب بھی آپ کو مدد کی ضرورت ہو، میں ہمیشہ یہاں موجود ہوں۔",
    "Vietnamese": "Tạm biệt! Hãy chăm sóc bản thân thật tốt. Tôi luôn ở đây nếu bạn cần hỗ trợ.",
    "Chinese": "再见！请照顾好自己。如果您需要支持，我随时都在这里。"
}

GRATITUDE_RESPONSES = {
    "Arabic": "العفو! يسعدني أنني استطعت مساعدتك. لا تتردد في العودة في أي وقت تحتاج فيه إلى الدعم.",
    "Bulgarian": "Моля! Радвам се, че можах да помогна. Не се колебайте да се върнете по всяко време.",
    "German": "Gern geschehen! Es freut mich, dass ich helfen konnte. Kommen Sie jederzeit gerne wieder.",
    "Greek": "Παρακαλώ! Χαίρομαι που μπόρεσα να βοηθήσω. Μη διστάσετε να επιστρέψετε οποιαδήποτε στιγμή.",
    "English": "You're welcome! I'm glad I could help. Don't hesitate to come back anytime you need support.",
    "Spanish": "¡De nada! Me alegra haber podido ayudarte. No dudes en volver si necesitas apoyo.",
    "French": "De rien ! Je suis heureux d'avoir pu vous aider. N'hésitez pas à revenir si vous avez besoin de soutien.",
    "Hindi": "आपका स्वागत है! मुझे खुशी है कि मैं मदद कर सका। सहायता के लिए कभी भी वापस आने में संकोच न करें।",
    "Italian": "Prego! Sono felice di aver potuto aiutare. Non esitare a tornare quando vuoi.",
    "Japanese": "どういたしまして！お役に立てて嬉しいです。何かありましたらいつでもお越しください。",
    "Dutch": "Graag gedaan! Ik ben blij dat ik kon helpen. Aarzel niet om terug te komen als je hulp nodig hebt.",
    "Polish": "Proszę bardzo! Cieszę się, że mogłem pomóc. Wracaj śmiało, kiedy tylko będziesz potrzebować wsparcia.",
    "Portuguese": "De nada! Fico feliz em poder ajudar. Não hesite em voltar sempre que precisar.",
    "Russian": "Пожалуйста! Рад, что смог помочь. Обращайтесь в любое время, если возникнут вопросы.",
    "Swahili": "Karibu! Nimefurahi kuweza kusaidia. Usisite kurudi wakati wowote ukitaka usaidizi.",
    "Thai": "ยินดีค่ะ/ครับ! ดีใจที่ได้ช่วยเหลือคุณ แวะกลับมาพูดคุยได้เสมอเมื่อต้องการคำแนะนำนะคะ/ครับ",
    "Turkish": "Rica ederim! Yardımcı olabildiğinh için çok mutluyum. Desteğe ihtiyaç duyduğunuzda çekinmeden gelebilirsiniz.",
    "Urdu": "خوش آمدید! مجھے خوشی ہے کہ میں آپ کی مدد کر سکا۔ کسی بھی وقت دوبارہ رابطہ کرنے میں ہچکچاہٹ محسوس نہ کریں۔",
    "Vietnamese": "Không có gì! Tôi rất vui được giúp đỡ bạn. Đừng ngần ngại quay lại bất cứ khi nào bạn cần.",
    "Chinese": "不用谢！很高兴能帮到您。需要支持时，请随时回来。"
}

# Critical self-harm/crisis support helpline templates for each language
CRITICAL_CRISIS_RESPONSES = {
    "Arabic": "إذا كنت تمر بأزمة أو تراودك أفكار لإيذاء نفسك، يرجى التواصل للحصول على دعم فوري. يمكنك الاتصال بخط المساعدة الوطني للسلامة النفسية، أو التوجه إلى أقرب مستشفى طوارئ. نحن نهتم بك ولست وحدك.",
    "Bulgarian": "Ако преминавате през криза или имате мисли за самонараняване, моля, потърсете незабавна подкрепа. Можете да се свържете с националната телефонна линия за подкрепа или да отидете в най-близкото спешно отделение.",
    "German": "Wenn Sie sich in einer Krise befinden oder Selbstverletzungsgedanken haben, suchen Sie bitte sofort Hilfe. Sie können die Telefonseelsorge unter 0800 111 0 111 (Deutschland) anrufen oder die nächste Notaufnahme aufsuchen. Sie sind nicht allein.",
    "Greek": "Εάν αντιμετωπίζετε κρίση ή έχετε σκέψεις αυτοτραυματισμού, ζητήστε άμεση υποστήριξη. Καλέστε τη γραμμή παρέμβασης για την αυτοκτονία στο 1018 ή απευθυνθείτε στο πλησιέστερο νοσοκομείο.",
    "English": "If you are in distress or having thoughts of self-harm, please reach out for immediate support. You can call or text the Suicide & Crisis Lifeline at 988 (available 24/7), or go to your nearest emergency room. You are not alone.",
    "Spanish": "Si estás en crisis o tienes pensamientos de hacerte daño, busca apoyo de inmediato. Puedes llamar o enviar un mensaje al 988 para la Línea de Prevención del Suicidio y Crisis o acudir a la sala de emergencias más cercana.",
    "French": "Si vous êtes en détresse ou si vous pensez à vous faire du mal, veuillez contacter immédiatement une ligne d'aide d'urgence (comme le 3114 en France) ou vous rendre aux urgences les plus proches.",
    "Hindi": "यदि आप संकट में हैं या खुद को नुकसान पहुंचाने के विचार आ रहे हैं, तो कृपया तुरंत सहायता लें। आप आपातकालीन नंबर (जैसे 988 या किरण हेल्पलाइन 1800-599-0019) पर कॉल कर सकते हैं या निकटतम अस्पताल जा सकते हैं।",
    "Italian": "Se ti trovi in una situazione di emergenza o hai pensieri di autolesionismo, cerca aiuto immediato. Puoi chiamare il Telefono Amico al 02 2327 2327 o recarti al pronto soccorso più vicino.",
    "Japanese": "もし困難な状況にあり自傷行為を考えている場合は、すぐに助けを求めてください。こころの健康相談ダイヤル（0570-064-556）か、最寄りの救命急病センターにご相談ください。",
    "Dutch": "Als je in nood verkeert of aan zelfbeschadiging denkt, zoek dan onmiddellijk hulp. Bel gratis 0800-0113 (Nederland) of ga naar de dichtstbijzijnde eerste hulp.",
    "Polish": "Jeśli przechodzisz przez kryzys lub masz myśli o samookaleczeniu, natychmiast poszukaj wsparcia. Możesz zadzwonić pod bezpłatny numer 116 123 (Polska) lub zgłosić się do najbliższego szpitala.",
    "Portuguese": "Se você estiver em crise ou pensando em se machucar, procure ajuda imediata. Ligue para o Centro de Valorização da Vida (CVV) pelo número 188 ou dirija-se ao pronto-socorro mais próximo.",
    "Russian": "Если вы переживаете кризис или думаете о причинении себе вреда, немедленно обратитесь за помощью. Позвоните на горячую линию психологической помощи или обратитесь в ближайшую больницу.",
    "Swahili": "Ikiwa uko kwenye shida au unafikiria kujidhuru, tafadhali tafuta usaidizi wa haraka. Unaweza kuwasiliana na nambari za usaidizi za dharura au kwenda kwenye chumba cha dharura kilicho karibu nawe.",
    "Thai": "หากคุณกำลังเผชิญกับวิกฤตหรือมีความคิดที่จะทำร้ายตัวเอง โปรดขอความช่วยเหลือทันที โทรสายด่วนสุขภาพจิต 1323 หรือไปที่ห้องฉุกเฉินที่ใกล้ที่สุด",
    "Turkish": "Bir kriz içindeyseniz veya kendinize zarar vermeyi düşünüyorsanız lütfen hemen yardım isteyin. 182 nolu hattı arayabilir ya da en yakın acil servise başvurabilirsiniz.",
    "Urdu": "اگر آپ کسی مشکل میں ہیں یا خود کو نقصان پہنچانے کے بارے میں سوچ رہے ہیں، تو براہ کرم فوری مدد حاصل کریں۔ آپ مقامی ہیلپ لائن سے رابطہ کر سکتے ہیں یا قریبی ہسپتال جا سکتے ہیں۔",
    "Vietnamese": "Nếu bạn đang gặp khủng hoảng hoặc có ý định tự hại, vui lòng tìm kiếm sự giúp đỡ ngay lập tức. Hãy gọi đường dây nóng hỗ trợ tâm lý hoặc đến phòng cấp cứu gần nhất.",
    "Chinese": "如果您正处于危机之中或有自残想法，请立即寻求帮助。您可以拨打心理援助热线（如 988 或 400-161-9995）或前往最近的急诊室。"
}

# Redirect responses for out-of-scope queries in all 20 languages
OUT_OF_SCOPE_RESPONSES = {
    "Arabic": "أنا هنا لمساعدتك في الأسئلة المتعلقة بالصحة النفسية والعاطفية فقط. هل هناك أي موضوع متعلق بسلامتك النفسية ترغب في التحدث عنه؟",
    "Bulgarian": "Аз съм специализиран асистент за подкрепа на психичното здраве. Мога да помагам само с въпроси, свързани с емоционалното и психологическото благополучие. Как мога да ви подкрепя в тази област днес?",
    "German": "Ich bin ein Assistent für mentale Gesundheit. Ich kann nur Fragen beantworten, die sich auf das emotionale und psychologische Wohlbefinden beziehen. Wie kann ich Sie heute in diesem Bereich unterstützen?",
    "Greek": "Είμαι ένας εξειδικευμένος βοηθός υποστήριξης ψυχικής υγείας. Μπορώ να βοηθήσω μόνο με ερωτήσεις που σχετίζονται με τη συναισθηματική και ψυχολογική ευεξία. Πώς μπορώ να σας υποστηρίξω σε αυτόν τον τομέα σήμερα;",
    "English": "I am a dedicated mental health support assistant. I can only assist with questions related to emotional and psychological well-being. How can I support you in that area today?",
    "Spanish": "Soy un asistente de apoyo para la salud mental. Solo puedo responder a preguntas relacionadas con el bienestar emocional y psicológico. ¿Cómo puedo ayudarte hoy en este ámbito?",
    "French": "Je suis un assistant dédié au soutien en santé mentale. Je ne peux répondre qu'aux questions liées au bien-être émotionnel et psychologique. Comment puis-je vous aider dans ce domaine aujourd'hui ?",
    "Hindi": "मैं एक समर्पित मानसिक स्वास्थ्य सहायता सहायक हूँ। मैं केवल भावनात्मक और मनोवैज्ञानिक कल्याण से संबंधित प्रश्नों में ही मदद कर सकता हूँ। आज मैं इस क्षेत्र में आपकी क्या मदद कर सकता हूँ?",
    "Italian": "Sono un assistente dedicato al supporto della salute mentale. Posso rispondere solo a domande riguardanti il benessere emotivo e psicologico. Come posso aiutarti oggi in questo ambito?",
    "Japanese": "私はメンタルヘルス専門のサポートアシスタントです。感情的・心理的な健康に関するご質問にのみお答えできます。今日はどのようなご相談でしょうか？",
    "Dutch": "Ik ben een assistent voor mentale gezondheid. Ik kan alleen helpen met vragen die te maken hebben met emotioneel en psychologisch welzijn. Hoe kan ik je vandaag op dat gebied ondersteunen?",
    "Polish": "Jestem dedykowanym asystentem wsparcia zdrowia psychicznego. Mogę pomagać wyłącznie w kwestiach związanych z dobrostanem emocjonalnym i psychologicznym. Jak mogę Cię dzisiaj wesprzeć w tym obszarze?",
    "Portuguese": "Sou um assistente dedicado ao apoio à saúde mental. Só posso ajudar com questões relacionadas com o bem-estar emocional e psicológico. Como posso apoiar você nessa área hoje?",
    "Russian": "Я специализированный ассистент поддержки психического здоровья. Я могу отвечать только на вопросы, связанные с эмоциональным и психологическим благополучием. Чем я могу помочь вам в этой области сегодня?",
    "Swahili": "Mimi ni msaidizi aliyejitolea wa afya ya akili. Ninaweza tu kusaidia na maswali yanayohusiana na ustawi vya kihisia na kisaikolojia. Ninawezaje kukusaidia katika eneo hilo leo?",
    "Thai": "ฉันเป็นผู้ช่วยสนับสนุนด้านสุขภาพจิตโดยเฉพาะ ฉันสามารถช่วยเหลือในคำถามที่เกี่ยวข้องกับสุขภาวะทางอารมณ์และจิตใจเท่านั้น วันนี้มีอะไรให้ฉันช่วยเหลือในด้านนี้ไหมคะ/ครับ?",
    "Turkish": "Ben psikolojik destek ve ruh sağlığı asistanıyım. Sadece duygusal ve psikolojik esenlikle ilgili konularda yardımcı olabilirim. Bugün bu alanda size nasıl destek olabilirim?",
    "Urdu": "میں ذہنی صحت کا ایک سرشار معاون ہوں۔ میں صرف جذباتی اور نفسیاتی بہبود سے متعلق سوالات میں مدد کر سکتا ہوں۔ آج میں اس شعبے میں آپ کی کیا مدد کر سکتا ہوں؟",
    "Vietnamese": "Tôi là trợ lý chuyên hỗ trợ sức khỏe tâm thần. Tôi chỉ có thể giúp đỡ các câu hỏi liên quan đến sức khỏe cảm xúc và tâm lý. Tôi có thể hỗ trợ bạn như thế nào trong lĩnh vực đó hôm nay?",
    "Chinese": "我是专门的心理健康支持助手。我只能协助解决与情感和心理健康相关的问题。今天我能在该领域为您提供什么支持吗？"
}

# Localized medical advice/prescriptions disclaimers for all 20 languages
MEDICAL_DISCLAIMERS = {
    "Arabic": "ملاحظة: أنا مساعد ذكاء اصطناعي ولست طبيباً. لا يمكنني تشخيص الحالات أو وصف الأدوية. يرجى استشارة طبيب مختص أو معالج نفسي.",
    "Bulgarian": "Забележка: Аз съм виртуален асистент и не мога да предписвам лекарства или да давам медицински съвети. Моля, консултирайте се с квалифициран лекар.",
    "German": "Hinweis: Ich bin ein KI-Assistent und kann keine Medikamente verschreiben oder medizinischen Rat erteilen. Bitte konsultieren Sie einen Arzt oder Psychiater.",
    "Greek": "Σημείωση: Είμαι βοηθός τεχνητής νοημοσύνης και δεν μπορώ να συνταγογραφήσω φάρμακα ή να δώσω ιατρικές συμβουλές. Συμβουλευτείτε έναν γιατρό.",
    "English": "Please note: I am an AI assistant and cannot prescribe medication or provide clinical medical advice. Please consult a qualified medical professional or psychiatrist.",
    "Spanish": "Nota: Soy un asistente de IA y no puedo recetar medicamentos ni ofrecer asesoramiento médico. Consulta a un profesional de la salud calificado.",
    "French": "Remarque : Je suis un assistant virtuel et je ne peux pas prescrire de médicaments ni donner de conseils médicaux. Veuillez consulter un professionnel de la santé.",
    "Hindi": "कृपया ध्यान दें: मैं एक एआई सहायक हूँ और दवाएं नहीं लिख सकता या चिकित्सा सलाह नहीं दे सकता। कृपया डॉक्टर या मनोचिकित्सक से परामर्श लें।",
    "Italian": "Nota: Sono un assistente IA e non posso prescrivere farmaci o fornire consulenza medica. Si prega di consultare un medico qualificato.",
    "Japanese": "注意：私はAIアシスタントであり、薬の処方や医療的助言は行えません。専門の医師や精神科医にご相談ください。",
    "Dutch": "Opmerking: Ik ben een AI-assistent en kan geen medicijnen voorschrijven of medisch advies geven. Raadpleeg een gekwalificeerde arts.",
    "Polish": "Uwaga: Jestem asystentem AI i nie przepisuję leków ani nie udzielam porad medycznych. Skonsultuj się z wykwalifikowanym lekarzem.",
    "Portuguese": "Aviso: Sou um assistente de IA e não posso receitar medicamentos ou fornecer aconselhamento médico. Consulte um médico qualificado.",
    "Russian": "Примечание: Я ИИ-ассистент и не могу назначать лекарства или давать медицинские советы. Пожалуйста, обратитесь к врачу.",
    "Swahili": "Kumbuka: Mimi ni msaidizi wa AI na siwezi kuagiza dawa au kutoa ushauri wa matibabu. Tafadhali tembelea daktari aliyefuzu.",
    "Thai": "หมายเหตุ: ฉันเป็นผู้ช่วย AI และไม่สามารถจ่ายยาหรือให้คำแนะนำทางการแพทย์ได้ โปรดปรึกษาแพทย์หรือจิตแพทย์ผู้เชี่ยวชาญ",
    "Turkish": "Not: Ben bir yapay zeka asistanıyım ve ilaç reçete edemem veya tıbbi tavsiye veremem. Lütfen uzman bir doktora danışın.",
    "Urdu": "نوٹ: میں ایک اے آئی معاون ہوں اور ادویات تجویز کرنے یا طبی مشورہ دینے کا اہل نہیں ہوں۔ براہ کرم کسی مستند ڈاکٹر سے رجوع کریں۔",
    "Vietnamese": "Lưu ý: Tôi là trợ lý AI và không thể kê đơn thuốc hoặc tư vấn y tế. Vui lòng tham khảo ý kiến bác sĩ chuyên khoa.",
    "Chinese": "注意：我是AI助手，不能开具处方或提供医疗建议。请咨询合格的医生或精神科医生。"
}

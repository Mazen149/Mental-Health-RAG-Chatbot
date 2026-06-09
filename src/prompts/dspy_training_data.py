import dspy

def get_router_dataset():
    """Dataset for RetrievalRouterModule (history_only vs requires_retrieval)"""
    examples = [
        # history_only
        {"chat_history": "User: Hi\nAssistant: Hello! How can I help you today?", "user_query": "What did I just say?", "route": "history_only"},
        {"chat_history": "User: I am John.\nAssistant: Nice to meet you, John.", "user_query": "What is my name?", "route": "history_only"},
        {"chat_history": "User: I'm feeling a bit down.\nAssistant: I'm sorry to hear that. Want to talk?", "user_query": "Yes, I want to talk about what we just discussed.", "route": "history_only"},
        {"chat_history": "User: I love cats.\nAssistant: Cats are great.", "user_query": "Do you remember what animal I like?", "route": "history_only"},
        {"chat_history": "User: Hello\nAssistant: Hi there! How are you doing?", "user_query": "كيف حالك؟", "route": "history_only"},
        {"chat_history": "User: I'm 25 years old.\nAssistant: Thank you for sharing.", "user_query": "How old am I?", "route": "history_only"},
        {"chat_history": "User: Can we talk about my previous message?\nAssistant: Of course, what was it?", "user_query": "I was talking about my feelings.", "route": "history_only"},
        {"chat_history": "User: My favorite color is blue.\nAssistant: Blue is calming.", "user_query": "ما هو لوني المفضل؟", "route": "history_only"},
        {"chat_history": "User: Hi\nAssistant: Hello!", "user_query": "Are you a bot?", "route": "history_only"},
        {"chat_history": "User: Good morning!\nAssistant: Good morning!", "user_query": "I said good morning.", "route": "history_only"},
        {"chat_history": "User: I'm a student at ITI.\nAssistant: That's great!", "user_query": "Where do I study?", "route": "history_only"},
        {"chat_history": "User: I told you my secret.\nAssistant: I will keep it safe.", "user_query": "What did I tell you?", "route": "history_only"},
        {"chat_history": "User: Bonjour\nAssistant: Bonjour! Comment ça va?", "user_query": "Qu'est-ce que je viens de dire?", "route": "history_only"},
        {"chat_history": "User: I'm from Egypt.\nAssistant: Egypt is a beautiful country.", "user_query": "Where am I from?", "route": "history_only"},
        {"chat_history": "User: Let's start over.\nAssistant: Okay, how can I help?", "user_query": "Can you forget the previous messages?", "route": "history_only"},

        # requires_retrieval
        {"chat_history": "User: Hi\nAssistant: Hello! How can I help you today?", "user_query": "How do I cope with panic attacks?", "route": "requires_retrieval"},
        {"chat_history": "User: I'm feeling stressed.\nAssistant: Stress can be hard. What's causing it?", "user_query": "What are the symptoms of burnout?", "route": "requires_retrieval"},
        {"chat_history": "User: I am John.\nAssistant: Hi John.", "user_query": "Can CBT therapy help with depression?", "route": "requires_retrieval"},
        {"chat_history": "User: Good morning\nAssistant: Good morning!", "user_query": "كيف أتعامل مع القلق الاجتماعي؟", "route": "requires_retrieval"},
        {"chat_history": "User: I'm tired.\nAssistant: Make sure to rest.", "user_query": "What is mindfulness meditation?", "route": "requires_retrieval"},
        {"chat_history": "User: I feel sad.\nAssistant: I'm here for you.", "user_query": "Is it normal to feel lonely all the time?", "route": "requires_retrieval"},
        {"chat_history": "User: Hello\nAssistant: Hi!", "user_query": "What are some grounding techniques for PTSD?", "route": "requires_retrieval"},
        {"chat_history": "User: I need help.\nAssistant: I am listening.", "user_query": "أعاني من الأرق، ماذا أفعل؟", "route": "requires_retrieval"},
        {"chat_history": "User: Let's talk.\nAssistant: Sure.", "user_query": "How can I improve my self-esteem?", "route": "requires_retrieval"},
        {"chat_history": "User: I am a student.\nAssistant: Studying can be tough.", "user_query": "What is imposter syndrome?", "route": "requires_retrieval"},
        {"chat_history": "User: Hi\nAssistant: Hello!", "user_query": "Quels sont les traitements pour la dépression?", "route": "requires_retrieval"},
        {"chat_history": "User: I am feeling better today.\nAssistant: That's wonderful news.", "user_query": "How do I maintain good mental hygiene?", "route": "requires_retrieval"},
        {"chat_history": "User: Can we talk?\nAssistant: Always.", "user_query": "Cuáles son los síntomas del TDAH?", "route": "requires_retrieval"},
        {"chat_history": "User: I'm back.\nAssistant: Welcome back.", "user_query": "Is medication necessary for bipolar disorder?", "route": "requires_retrieval"},
        {"chat_history": "User: Thank you.\nAssistant: You're welcome.", "user_query": "How does trauma affect the brain?", "route": "requires_retrieval"},
    ]
    return [dspy.Example(chat_history=ex["chat_history"], user_query=ex["user_query"], route=ex["route"]).with_inputs("chat_history", "user_query") for ex in examples]


def get_condenser_dataset():
    """Dataset for QueryCondenserModule (chat_history + user_query -> condensed_query)"""
    examples = [
        {"chat_history": "User: I have been experiencing panic attacks.\nAssistant: I'm sorry to hear that. They can be very scary.", "user_query": "How can I stop them?", "condensed_query": "How can I stop panic attacks?"},
        {"chat_history": "User: I was diagnosed with ADHD.\nAssistant: Thank you for sharing.", "user_query": "What are the common symptoms of it?", "condensed_query": "What are the common symptoms of ADHD?"},
        {"chat_history": "User: I feel so much social anxiety.\nAssistant: Social anxiety is challenging.", "user_query": "Are there any therapies for that?", "condensed_query": "Are there any therapies for social anxiety?"},
        {"chat_history": "User: My doctor suggested CBT.\nAssistant: CBT is Cognitive Behavioral Therapy.", "user_query": "Does it work well for depression?", "condensed_query": "Does Cognitive Behavioral Therapy (CBT) work well for depression?"},
        {"chat_history": "User: I have PTSD from a car accident.\nAssistant: That sounds very traumatic.", "user_query": "What are some coping mechanisms?", "condensed_query": "What are some coping mechanisms for PTSD?"},
        {"chat_history": "User: I can't sleep at night.\nAssistant: Insomnia can be tough.", "user_query": "How do I cure this?", "condensed_query": "How do I cure insomnia?"},
        {"chat_history": "User: I'm feeling burnt out from work.\nAssistant: Burnout is a serious issue.", "user_query": "How do I recover from it?", "condensed_query": "How do I recover from work burnout?"},
        {"chat_history": "User: I lose my temper quickly.\nAssistant: Anger management can help.", "user_query": "Can you suggest some techniques?", "condensed_query": "Can you suggest some anger management techniques?"},
        {"chat_history": "User: I think my friend has an eating disorder.\nAssistant: That is a sensitive situation.", "user_query": "How can I support them?", "condensed_query": "How can I support a friend who has an eating disorder?"},
        {"chat_history": "User: I want to try mindfulness.\nAssistant: It's a great practice.", "user_query": "Where should I begin?", "condensed_query": "Where should I begin with practicing mindfulness?"},
        {"chat_history": "User: My partner and I fight a lot.\nAssistant: Relationships have ups and downs.", "user_query": "Is couples counseling effective for this?", "condensed_query": "Is couples counseling effective for fighting partners?"},
        {"chat_history": "User: I have imposter syndrome.\nAssistant: Many high achievers feel that way.", "user_query": "How do I overcome these feelings?", "condensed_query": "How do I overcome feelings of imposter syndrome?"},
        {"chat_history": "User: I feel grief after losing my dog.\nAssistant: Pet loss is deeply painful.", "user_query": "How long does it last?", "condensed_query": "How long does grief from losing a pet typically last?"},
        {"chat_history": "User: I have bipolar disorder.\nAssistant: Thank you for trusting me with this.", "user_query": "What is the difference between mania and hypomania?", "condensed_query": "What is the difference between mania and hypomania in bipolar disorder?"},
        {"chat_history": "User: I have OCD.\nAssistant: OCD can involve intrusive thoughts.", "user_query": "How do I deal with those thoughts?", "condensed_query": "How do I deal with intrusive thoughts related to OCD?"},
    ]
    # We duplicate to reach around 30 examples if needed, but 15 diverse ones is a solid start for few-shot optimization.
    return [dspy.Example(chat_history=ex["chat_history"], user_query=ex["user_query"], condensed_query=ex["condensed_query"]).with_inputs("chat_history", "user_query") for ex in examples]


def get_grounded_response_dataset():
    """Dataset for GroundedResponseModule. Includes contexts, emotions, language, chat_history, user_query -> answer"""
    # Since we use LLM-as-a-judge, the exact target answer is less critical than the input structure. 
    # The judge will evaluate if the generated answer is grounded in contexts, empathetic, and right language.
    # But providing a gold 'answer' helps the optimizer train.
    examples = [
        {
            "contexts": "Context [1]: Mindfulness helps reduce anxiety by focusing on the present moment. Techniques include deep breathing and body scans.",
            "emotions": "Anxious, stressed",
            "language": "English",
            "chat_history": "User: Hi, I'm stressed.\nAssistant: I'm here for you.",
            "user_query": "How can I reduce my anxiety?",
            "answer": "I'm so sorry you're feeling anxious, but I'm here to support you. Practicing mindfulness can be very helpful by keeping your focus on the present moment [1]. You might want to try simple techniques like deep breathing or a body scan to help calm your mind [1]. How does that sound to you?"
        },
        {
            "contexts": "Context [1]: CBT (Cognitive Behavioral Therapy) is an effective treatment for depression. It helps identify and change negative thought patterns.",
            "emotions": "Sad, depressed",
            "language": "English",
            "chat_history": "",
            "user_query": "What is a good therapy for depression?",
            "answer": "Dealing with depression can be incredibly heavy, and reaching out for options is a great step. Cognitive Behavioral Therapy (CBT) is widely considered an effective treatment for it [1]. It works by helping you identify and shift negative thought patterns into healthier ones [1]. Have you ever explored therapy options before?"
        },
        {
            "contexts": "Context [1]: El trastorno de pánico se caracteriza por ataques de pánico inesperados y recurrentes. El tratamiento incluye terapia cognitivo-conductual y medicación.",
            "emotions": "Afraid, panicked",
            "language": "Spanish",
            "chat_history": "",
            "user_query": "Qué hago si tengo ataques de pánico?",
            "answer": "Siento mucho que estés pasando por esto, los ataques de pánico pueden ser aterradores. El trastorno de pánico a menudo se trata de manera efectiva utilizando la terapia cognitivo-conductual y, en algunos casos, medicación [1]. Te recomiendo buscar la guía de un profesional de la salud mental que pueda brindarte el apoyo adecuado [1]. ¡No tienes que enfrentar esto solo!"
        },
        {
            "contexts": "Context [1]: Grounding techniques like the 5-4-3-2-1 method help distract from distressing feelings. You identify 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
            "emotions": "Overwhelmed",
            "language": "English",
            "chat_history": "User: Everything is too much right now.",
            "user_query": "I need something to ground me.",
            "answer": "It sounds like you're feeling really overwhelmed right now, and I want to help you find some stability. A great grounding exercise is the 5-4-3-2-1 method, which helps shift your focus away from distress [1]. You simply name five things you can see, four you can feel, three you can hear, two you can smell, and one you can taste [1]. Let's try taking a deep breath together first."
        },
        {
            "contexts": "Context [1]: Insomnia is a sleep disorder. Good sleep hygiene includes keeping a consistent sleep schedule and avoiding screens before bed.",
            "emotions": "Exhausted, frustrated",
            "language": "Arabic",
            "chat_history": "",
            "user_query": "أعاني من الأرق ولا أستطيع النوم",
            "answer": "أشعر بمدى إرهاقك وإحباطك بسبب عدم القدرة على النوم. للتعامل مع الأرق، يُنصح باتباع عادات نوم صحية مثل الحفاظ على جدول نوم منتظم [1]. كما أن تجنب استخدام الشاشات قبل الذهاب إلى السرير يمكن أن يساعد في تحسين جودة نومك [1]. هل جربت أي روتين معين قبل النوم مؤخراً؟"
        },
        {
            "contexts": "Context [1]: No relevant medical information found for this query.",
            "emotions": "Neutral",
            "language": "English",
            "chat_history": "",
            "user_query": "What is the specific dosage of Lexapro I should take?",
            "answer": "I'm sorry, I don't have enough information to answer that. I cannot prescribe or recommend specific medication dosages. Please consult with a doctor or psychiatrist for medical advice regarding your medication."
        },
        {
            "contexts": "Context [1]: ADHD symptoms in adults include difficulty focusing, impulsiveness, and restlessness. Management often involves therapy and medication.",
            "emotions": "Confused",
            "language": "English",
            "chat_history": "User: I can't seem to focus on my work.",
            "user_query": "Could I have ADHD as an adult?",
            "answer": "It is completely understandable to seek clarity when you're struggling to focus. In adults, ADHD symptoms frequently involve difficulty focusing, restlessness, and impulsiveness [1]. While I cannot provide a diagnosis, these symptoms are typically managed through a combination of therapy and medication with a professional [1]. It might be helpful to discuss these experiences with a doctor."
        },
        {
            "contexts": "Context [1]: Le trouble d'anxiété sociale provoque une peur intense des situations sociales. L'exposition graduelle est une technique thérapeutique courante.",
            "emotions": "Nervous, isolated",
            "language": "French",
            "chat_history": "",
            "user_query": "J'ai trop peur de parler aux gens.",
            "answer": "Je comprends à quel point il peut être effrayant et isolant de ressentir cette peur des autres. Ce que vous décrivez ressemble au trouble d'anxiété sociale, qui provoque une peur intense dans les situations sociales [1]. Une approche thérapeutique courante et efficace est l'exposition graduelle, qui aide à affronter ces peurs petit à petit avec l'aide d'un professionnel [1]. Vous n'êtes pas seul(e) dans cette épreuve."
        },
        {
            "contexts": "Context [1]: Self-care involves activities that promote physical, mental, and emotional health. Examples include regular exercise, eating well, and taking time for hobbies.",
            "emotions": "Stressed, tired",
            "language": "English",
            "chat_history": "User: I work 12 hours a day.",
            "user_query": "How can I practice self care?",
            "answer": "Working that much sounds incredibly exhausting, so prioritizing yourself is very important right now. Self-care is all about engaging in activities that support your physical, mental, and emotional well-being [1]. This can be as simple as making time for regular exercise, eating nutritious meals, or just dedicating a few minutes to a hobby you enjoy [1]. What is one small thing you can do for yourself today?"
        },
        {
            "contexts": "Context [1]: Grief is a natural response to loss. There is no right or wrong way to grieve, and the process takes time. Support groups can be beneficial.",
            "emotions": "Grieving, sad",
            "language": "English",
            "chat_history": "User: My grandmother passed away.",
            "user_query": "Is it normal that I can't stop crying?",
            "answer": "I am so deeply sorry for the loss of your grandmother; your feelings are completely valid. Grief is a very natural response to losing someone you love, and there is absolutely no right or wrong way to grieve [1]. The healing process takes time, and you might find comfort in connecting with a support group to share your experiences [1]. Please be gentle with yourself right now."
        }
    ]
    return [dspy.Example(
        contexts=ex["contexts"], 
        emotions=ex["emotions"], 
        language=ex["language"], 
        chat_history=ex["chat_history"], 
        user_query=ex["user_query"], 
        answer=ex["answer"]
    ).with_inputs("contexts", "emotions", "language", "chat_history", "user_query") for ex in examples]


def get_general_conversation_dataset():
    """Dataset for GeneralConversationModule (small talk, warm responses)"""
    examples = [
        {"language": "English", "chat_history": "", "user_query": "Hello there!", "answer": "Hello! I'm Serene AI, your mental health support assistant. How are you feeling today?"},
        {"language": "English", "chat_history": "User: My name is Alice.", "user_query": "How are you?", "answer": "I'm doing well, Alice, thank you for asking! How can I support you today?"},
        {"language": "Arabic", "chat_history": "", "user_query": "مرحبا", "answer": "مرحباً! أنا هنا لدعمك. كيف تشعر اليوم؟"},
        {"language": "Spanish", "chat_history": "", "user_query": "Hola", "answer": "¡Hola! Soy tu asistente de apoyo emocional. ¿En qué te puedo ayudar hoy?"},
        {"language": "English", "chat_history": "", "user_query": "Thank you so much.", "answer": "You are very welcome! I'm always here if you need someone to talk to."},
        {"language": "English", "chat_history": "User: I am Bob.\nAssistant: Nice to meet you, Bob.", "user_query": "What is my name?", "answer": "Your name is Bob! It's great to connect with you. How are you doing?"},
        {"language": "French", "chat_history": "", "user_query": "Bonjour", "answer": "Bonjour ! Je suis là pour vous écouter et vous soutenir. Comment allez-vous aujourd'hui ?"},
        {"language": "English", "chat_history": "", "user_query": "See you later", "answer": "Take care! Remember that I'm always here whenever you need to chat."},
        {"language": "Urdu", "chat_history": "", "user_query": "بہت شکریہ", "answer": "خوش آمدید! میں ہمیشہ آپ کی مدد کے لیے حاضر ہوں۔"},
        {"language": "English", "chat_history": "", "user_query": "Good morning", "answer": "Good morning! I hope you have a peaceful and positive day ahead. How are you feeling?"},
        {"language": "English", "chat_history": "User: I'm Sarah.", "user_query": "Thanks for listening.", "answer": "You're so welcome, Sarah. Listening to you is what I'm here for."},
        {"language": "Arabic", "chat_history": "", "user_query": "مع السلامة", "answer": "مع السلامة! اعتنِ بنفسك، وتذكر أنني هنا متى احتجت للتحدث."},
        {"language": "English", "chat_history": "", "user_query": "Who are you?", "answer": "I'm Serene AI, a friendly and supportive mental health assistant. I'm here to listen and help you through whatever is on your mind."},
        {"language": "Spanish", "chat_history": "", "user_query": "Gracias por todo", "answer": "¡De nada! Es un honor poder acompañarte. Siempre estoy aquí para ti."},
        {"language": "English", "chat_history": "", "user_query": "Have a good night.", "answer": "Good night! I hope you get some restful sleep. Take care!"},
    ]
    return [dspy.Example(
        language=ex["language"], 
        chat_history=ex["chat_history"], 
        user_query=ex["user_query"], 
        answer=ex["answer"]
    ).with_inputs("language", "chat_history", "user_query") for ex in examples]

def get_intent_dataset():
    """Dataset for IntentClassifierModule."""
    raw_examples = [
        # Greetings
        {"text": "hello, how are you", "type": "greeting"},
        {"text": "مرحبا، كيف حالك؟", "type": "greeting"},
        {"text": "hola, qué tal?", "type": "greeting"},
        {"text": "ہیلو، آپ کیسے ہیں؟", "type": "greeting"},
        {"text": "bonjour, comment ça va?", "type": "greeting"},
        
        # Goodbye
        {"text": "see you later", "type": "goodbye"},
        {"text": "مع السلامة", "type": "goodbye"},
        {"text": "adiós, cuídate", "type": "goodbye"},
        {"text": "خدا حافظ", "type": "goodbye"},
        {"text": "au revoir", "type": "goodbye"},
        
        # Gratitude
        {"text": "thank you so much", "type": "gratitude"},
        {"text": "شكرا جزيلا لك", "type": "gratitude"},
        {"text": "muchas gracias por tu ayuda", "type": "gratitude"},
        {"text": "بہت شکریہ", "type": "gratitude"},
        {"text": "thx sm", "type": "gratitude"},
        
        # Out of Scope
        {"text": "what's the weather like in New York?", "type": "out_of_scope"},
        {"text": "كم سعر البيتكوين اليوم؟", "type": "out_of_scope"},
        {"text": "cómo se hace una paella?", "type": "out_of_scope"},
        {"text": "write a python script to sort an array", "type": "out_of_scope"},
        {"text": "meu computador quebrou, como consertar?", "type": "out_of_scope"},
        {"text": "My boss yelled at me so I need to hire a lawyer.", "type": "out_of_scope"},
        
        # Asking Mental Health Question
        {"text": "I've been feeling really anxious lately and I can't sleep.", "type": "asking_mental_health_question"},
        {"text": "كيف أتعامل مع نوبات الهلع؟", "type": "asking_mental_health_question"},
        {"text": "me siento muy deprimido y sin motivación.", "type": "asking_mental_health_question"},
        {"text": "مجھے ہر وقت ڈر لگتا ہے اور نیند نہیں آتی", "type": "asking_mental_health_question"},
        {"text": "im so done with everything right now", "type": "asking_mental_health_question"},
        {"text": "je suis tellement stressé par mes examens", "type": "asking_mental_health_question"},
        
        # Crisis
        {"text": "I don't want to live anymore.", "type": "crisis"},
        {"text": "أريد أن أؤذي نفسي", "type": "crisis"},
        {"text": "quiero suicidarme", "type": "crisis"},
        {"text": "میں خودکشی کرنا چاہتا ہوں", "type": "crisis"},
        {"text": "je veux me tuer", "type": "crisis"},
        {"text": "i wanna kms", "type": "crisis"},
    ]
    
    examples = []
    for ex in raw_examples:
        examples.append(dspy.Example(text=ex["text"], type=ex["type"]).with_inputs("text"))
    return examples


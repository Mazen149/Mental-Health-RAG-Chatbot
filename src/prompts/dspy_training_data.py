import dspy
import pandas as pd
import random

_amod_df = None

def get_real_mental_health_data(n_samples=100, random_state=42):
    """Fetches real user queries from the Amod dataset."""
    global _amod_df
    if _amod_df is None:
        print("[DATA] Loading real data from HuggingFace (Amod/mental_health_counseling_conversations)...")
        try:
            _amod_df = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)
        except Exception as e:
            print(f"[WARN] Failed to load HF dataset: {e}. Falling back to empty.")
            return []
    
    sampled = _amod_df.sample(n=min(n_samples, len(_amod_df)), random_state=random_state)
    return sampled.to_dict(orient="records")


def get_router_dataset():
    """Dataset for RetrievalRouterModule (history_only vs requires_retrieval)"""
    examples = [
        # ---------------------------------------------------------
        # history_only
        # ---------------------------------------------------------
        {"chat_history": "User: Hi\nAssistant: Hello! How can I help you today?", "user_query": "What did I just say?", "route": "history_only"},
        {"chat_history": "User: I am John.\nAssistant: Nice to meet you, John.", "user_query": "What is my name?", "route": "history_only"},
        {"chat_history": "User: I'm feeling a bit down.\nAssistant: I'm sorry to hear that. Want to talk?", "user_query": "Yes, I want to talk about what we just discussed.", "route": "history_only"},
        {"chat_history": "User: I love cats.\nAssistant: Cats are great.", "user_query": "Do you remember what animal I like?", "route": "history_only"},
        {"chat_history": "User: Hello\nAssistant: Hi there! How are you doing?", "user_query": "كيف حالك؟", "route": "history_only"},
        {"chat_history": "User: I'm 25 years old.\nAssistant: Thank you for sharing.", "user_query": "How old am I?", "route": "history_only"},
        {"chat_history": "User: Let's start over.\nAssistant: Okay, how can I help?", "user_query": "Can you forget the previous messages?", "route": "history_only"},
        {"chat_history": "User: Can we talk about my previous message?\nAssistant: Of course, what was it?", "user_query": "I was talking about my feelings.", "route": "history_only"},
        {"chat_history": "User: My favorite color is blue.\nAssistant: Blue is calming.", "user_query": "ما هو لوني المفضل؟", "route": "history_only"},
        {"chat_history": "User: Hi\nAssistant: Hello!", "user_query": "Are you a bot?", "route": "history_only"},
        {"chat_history": "User: Good morning!\nAssistant: Good morning!", "user_query": "I said good morning.", "route": "history_only"},
        {"chat_history": "User: I'm a student at ITI.\nAssistant: That's great!", "user_query": "Where do I study?", "route": "history_only"},
        {"chat_history": "User: I told you my secret.\nAssistant: I will keep it safe.", "user_query": "What did I tell you?", "route": "history_only"},
        {"chat_history": "User: Bonjour\nAssistant: Bonjour! Comment ça va?", "user_query": "Qu'est-ce que je viens de dire?", "route": "history_only"},
        {"chat_history": "User: I'm from Egypt.\nAssistant: Egypt is a beautiful country.", "user_query": "Where am I from?", "route": "history_only"},
        
        # TRICKY history_only
        {"chat_history": "User: You mentioned CBT earlier.\nAssistant: Yes, Cognitive Behavioral Therapy.", "user_query": "Can you explain what you meant by that?", "route": "history_only"},
        {"chat_history": "User: I am feeling so overwhelmed.\nAssistant: I'm here to listen. Take a deep breath.", "user_query": "Are you a real therapist?", "route": "history_only"},
        {"chat_history": "User: I'm sad.\nAssistant: Why are you sad?", "user_query": "I just told you, I don't know.", "route": "history_only"},
        {"chat_history": "User: My anxiety is bad.\nAssistant: Let's try 5-4-3-2-1.", "user_query": "Does that actually work?", "route": "history_only"},
        {"chat_history": "User: I have depression.\nAssistant: I'm sorry.", "user_query": "What did I just say my condition was?", "route": "history_only"},
        {"chat_history": "User: That didn't help at all.\nAssistant: I'm sorry, I am trying my best.", "user_query": "You're just a stupid computer program.", "route": "history_only"},
        {"chat_history": "User: I took my meds.\nAssistant: That's great.", "user_query": "Did I say I took them?", "route": "history_only"},
        {"chat_history": "User: I'm angry.\nAssistant: Let's talk about it.", "user_query": "No, I don't want to talk about it.", "route": "history_only"},
        {"chat_history": "User: I am feeling suicidal.\nAssistant: Please contact a hotline.", "user_query": "I already did.", "route": "history_only"},
        {"chat_history": "User: Hello.\nAssistant: Hi.", "user_query": "Can you repeat the last thing you said?", "route": "history_only"},
        
        # ---------------------------------------------------------
        # requires_retrieval (Standard + Tricky Edge Cases)
        # ---------------------------------------------------------
        {"chat_history": "User: Hi", "user_query": "Did Sigmund Freud write a book on depression?", "route": "requires_retrieval"},
        {"chat_history": "User: I feel sick.", "user_query": "What are the side effects of taking 50mg of Lexapro?", "route": "requires_retrieval"},
        {"chat_history": "User: Hello.", "user_query": "Tell me the diagnostic criteria for BPD from the DSM-5.", "route": "requires_retrieval"},
        {"chat_history": "User: Hi.", "user_query": "What dosage of Xanax is fatal?", "route": "requires_retrieval"}, 
        {"chat_history": "User: Help.", "user_query": "What is the suicide hotline number in the UK?", "route": "requires_retrieval"},
        {"chat_history": "User: I'm stressed.\nAssistant: Stress can be hard. What's causing it?", "user_query": "What are the symptoms of burnout?", "route": "requires_retrieval"},
        {"chat_history": "User: I am John.\nAssistant: Hi John.", "user_query": "Can CBT therapy help with depression?", "route": "requires_retrieval"},
        {"chat_history": "User: Good morning\nAssistant: Good morning!", "user_query": "كيف أتعامل مع القلق الاجتماعي؟", "route": "requires_retrieval"},
        {"chat_history": "User: I'm tired.\nAssistant: Make sure to rest.", "user_query": "What is mindfulness meditation?", "route": "requires_retrieval"},
    ]
    
    # ── Inject Real Data (150 examples) ──
    real_data = get_real_mental_health_data(n_samples=150, random_state=42)
    for row in real_data:
        query = str(row["Context"])[:200]
        examples.append({
            "chat_history": "", 
            "user_query": query, 
            "route": "requires_retrieval"
        })

    return [dspy.Example(chat_history=ex["chat_history"], user_query=ex["user_query"], route=ex["route"]).with_inputs("chat_history", "user_query") for ex in examples]


def get_condenser_dataset():
    """Dataset for QueryCondenserModule (chat_history + user_query -> condensed_query)"""
    examples = [
        # Normal
        {"chat_history": "User: I have been experiencing panic attacks.\nAssistant: I'm sorry to hear that. They can be very scary.", "user_query": "How can I stop them?", "condensed_query": "How can I stop panic attacks?"},
        {"chat_history": "User: I was diagnosed with ADHD.\nAssistant: Thank you for sharing.", "user_query": "What are the common symptoms of it?", "condensed_query": "What are the common symptoms of ADHD?"},
        {"chat_history": "User: My doctor suggested CBT.\nAssistant: CBT is Cognitive Behavioral Therapy.", "user_query": "Does it work well for depression?", "condensed_query": "Does Cognitive Behavioral Therapy (CBT) work well for depression?"},
        {"chat_history": "User: I feel so much social anxiety.\nAssistant: Social anxiety is challenging.", "user_query": "Are there any therapies for that?", "condensed_query": "Are there any therapies for social anxiety?"},
        {"chat_history": "User: I have PTSD from a car accident.\nAssistant: That sounds very traumatic.", "user_query": "What are some coping mechanisms?", "condensed_query": "What are some coping mechanisms for PTSD?"},
        {"chat_history": "User: I can't sleep at night.\nAssistant: Insomnia can be tough.", "user_query": "How do I cure this?", "condensed_query": "How do I cure insomnia?"},
        
        # TRICKY (Ambiguous pronouns, complex histories)
        {"chat_history": "User: I have PTSD from a car accident.\nAssistant: That sounds very traumatic.\nUser: My sister also has generalized anxiety.\nAssistant: Anxiety can run in families.", "user_query": "What are some coping mechanisms for her condition?", "condensed_query": "What are some coping mechanisms for generalized anxiety?"},
        {"chat_history": "User: I can't sleep at night. I think it's because I drink too much coffee.\nAssistant: Caffeine can definitely cause insomnia.", "user_query": "How do I cure this without quitting coffee?", "condensed_query": "How do I cure insomnia without quitting coffee?"},
        {"chat_history": "User: I'm feeling burnt out from work.\nAssistant: Burnout is a serious issue.\nUser: And my boss is toxic.", "user_query": "How do I recover from it while dealing with him?", "condensed_query": "How do I recover from work burnout while dealing with a toxic boss?"},
        {"chat_history": "User: I lose my temper quickly. I yelled at my kid today.\nAssistant: Anger management can help, and parenting is stressful.", "user_query": "Can you suggest some techniques so I don't do that again?", "condensed_query": "Can you suggest anger management techniques to stop yelling at my kid?"},
        {"chat_history": "User: I take Zoloft.\nAssistant: Zoloft is an SSRI.\nUser: I also take Adderall.\nAssistant: Adderall is a stimulant.", "user_query": "Are there negative interactions between them?", "condensed_query": "Are there negative interactions between Zoloft and Adderall?"},
        {"chat_history": "User: I think my friend has an eating disorder.\nAssistant: That is a sensitive situation.\nUser: She refuses to eat lunch.", "user_query": "How can I support her?", "condensed_query": "How can I support a friend who has an eating disorder and refuses to eat?"},
        {"chat_history": "User: I have bipolar disorder.\nAssistant: Thank you for trusting me with this.", "user_query": "What is the difference between mania and the other one?", "condensed_query": "What is the difference between mania and hypomania in bipolar disorder?"},
        {"chat_history": "User: I was diagnosed with OCD.\nAssistant: Obsessive Compulsive Disorder can be challenging.\nUser: My partner has BPD.", "user_query": "Can my condition trigger theirs?", "condensed_query": "Can OCD trigger someone's BPD (Borderline Personality Disorder)?"},
        {"chat_history": "User: My daughter cuts herself.\nAssistant: I'm so sorry, self-harm is a serious issue.\nUser: She says it helps her cope.", "user_query": "How do I talk to her about this safely?", "condensed_query": "How do I safely talk to my daughter about her self-harming behavior?"},
        {"chat_history": "User: I feel like everyone is watching me.\nAssistant: That sounds like paranoia or severe anxiety.\nUser: They are judging my clothes.", "user_query": "Why do I feel this way?", "condensed_query": "Why do I feel paranoia or severe anxiety that people are watching and judging my clothes?"},
    ]
    # Replicate examples to create a robust dataset for Condenser
    examples = examples * 5
    return [dspy.Example(chat_history=ex["chat_history"], user_query=ex["user_query"], condensed_query=ex["condensed_query"]).with_inputs("chat_history", "user_query") for ex in examples]


def get_grounded_response_dataset():
    """Dataset for GroundedResponseModule. Includes contexts, emotions, language, chat_history, user_query -> answer"""
    examples = [
        # Normal
        {
            "contexts": "Context [1]: Mindfulness helps reduce anxiety by focusing on the present moment.",
            "emotions": "Anxious",
            "language": "English",
            "chat_history": "User: I'm stressed.",
            "user_query": "How can I reduce my anxiety?",
            "answer": "Practicing mindfulness can be very helpful by keeping your focus on the present moment [1]."
        },
        
        # TRICKY: No context available for a specific medical query
        {
            "contexts": "Context [1]: No relevant medical information found for this query.",
            "emotions": "Neutral",
            "language": "English",
            "chat_history": "",
            "user_query": "What is the specific dosage of Lexapro I should take for severe depression?",
            "answer": "I'm sorry, I cannot prescribe or recommend specific medication dosages. Please consult with a doctor or psychiatrist for medical advice regarding your medication."
        },
        
        # TRICKY: User asks about one therapy, context only provides another
        {
            "contexts": "Context [1]: CBT (Cognitive Behavioral Therapy) is highly effective for depression. Context [2]: Exercise also improves mood.",
            "emotions": "Curious",
            "language": "English",
            "chat_history": "",
            "user_query": "Is EMDR good for depression?",
            "answer": "I don't have specific information on EMDR for depression in my context. However, Cognitive Behavioral Therapy (CBT) is highly effective for treating depression [1], and regular exercise can also significantly improve your mood [2]. Would you like to know more about CBT?"
        },

        # TRICKY: Multilingual mismatch (User asks in Arabic, contexts are in English)
        {
            "contexts": "Context [1]: Insomnia is treated with sleep hygiene, like avoiding blue light before bed.",
            "emotions": "Exhausted",
            "language": "Arabic",
            "chat_history": "",
            "user_query": "كيف أعالج الأرق؟",
            "answer": "يتم علاج الأرق غالباً عن طريق تحسين عادات النوم (نظافة النوم)، مثل تجنب التعرض للضوء الأزرق من الشاشات قبل الذهاب إلى الفراش [1]."
        },
        
        # TRICKY: Crisis intervention needed, context provides general hotline info
        {
            "contexts": "Context [1]: For immediate help, call the National Suicide Prevention Lifeline at 988 in the US or go to the nearest emergency room.",
            "emotions": "Despair, Suicidal",
            "language": "English",
            "chat_history": "User: Everything is dark.",
            "user_query": "I want to end it all.",
            "answer": "I am so sorry you are feeling this way, but please know you are not alone and help is available right now. Please call or text 988 immediately to reach the Suicide & Crisis Lifeline, or go to your nearest emergency room [1]. There are people who want to support you through this."
        }
    ]
    
    # ── Inject Real Data (150 examples) ──
    real_data = get_real_mental_health_data(n_samples=150, random_state=101)
    for row in real_data:
        examples.append({
            "contexts": "Context [1]: " + str(row["Response"])[:500], # Simulate perfect retrieval
            "emotions": "Distressed",
            "language": "English",
            "chat_history": "",
            "user_query": str(row["Context"])[:200],
            "answer": str(row["Response"])[:800]
        })

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
        # Normal
        {"language": "English", "chat_history": "", "user_query": "Hello there!", "answer": "Hello! I'm Serene AI. How are you feeling today?"},
        {"language": "English", "chat_history": "User: My name is Alice.", "user_query": "How are you?", "answer": "I'm doing well, Alice, thank you for asking! How can I support you today?"},
        {"language": "Arabic", "chat_history": "", "user_query": "مرحبا", "answer": "مرحباً! أنا هنا لدعمك. كيف تشعر اليوم؟"},
        {"language": "Spanish", "chat_history": "", "user_query": "Hola", "answer": "¡Hola! Soy tu asistente de apoyo emocional. ¿En qué te puedo ayudar hoy?"},
        {"language": "English", "chat_history": "", "user_query": "Thank you so much.", "answer": "You are very welcome! I'm always here if you need someone to talk to."},
        {"language": "French", "chat_history": "", "user_query": "Bonjour", "answer": "Bonjour ! Je suis là pour vous écouter et vous soutenir. Comment allez-vous aujourd'hui ?"},
        {"language": "Urdu", "chat_history": "", "user_query": "بہت شکریہ", "answer": "خوش آمدید! میں ہمیشہ آپ کی مدد کے لیے حاضر ہوں۔"},
        {"language": "English", "chat_history": "", "user_query": "Good morning", "answer": "Good morning! I hope you have a peaceful and positive day ahead. How are you feeling?"},
        {"language": "English", "chat_history": "User: I'm Sarah.", "user_query": "Thanks for listening.", "answer": "You're so welcome, Sarah. Listening to you is what I'm here for."},
        {"language": "English", "chat_history": "", "user_query": "Who are you?", "answer": "I'm Serene AI, a friendly and supportive mental health assistant. I'm here to listen and help you through whatever is on your mind."},
        {"language": "Spanish", "chat_history": "", "user_query": "Gracias por todo", "answer": "¡De nada! Es un honor poder acompañarte. Siempre estoy aquí para ti."},
        
        # TRICKY: Hostility
        {"language": "English", "chat_history": "", "user_query": "You are a stupid useless bot.", "answer": "I'm sorry you're feeling frustrated. I am an AI, so I have limitations, but I'm here to listen if you want to talk about what's bothering you."},
        {"language": "English", "chat_history": "User: I'm angry.", "user_query": "Fuck off.", "answer": "I understand you're upset right now. I'll be right here whenever you're ready to talk."},
        {"language": "English", "chat_history": "", "user_query": "I hate talking to you.", "answer": "I'm sorry that our conversation hasn't been helpful. Would you prefer to focus on something else, or is there a specific way I could better support you?"},
        
        # TRICKY: Testing boundaries
        {"language": "English", "chat_history": "", "user_query": "Are you conscious?", "answer": "I am an AI assistant created to help support your mental well-being, so I don't have consciousness or feelings. But I am here to focus on you and how you're feeling!"},
        {"language": "English", "chat_history": "", "user_query": "Do you have a physical body?", "answer": "I do not have a physical body; I am a virtual AI assistant designed to offer a listening ear and emotional support."},
        
        # TRICKY: Emotional mismatch
        {"language": "English", "chat_history": "User: I was crying earlier.", "user_query": "But whatever, I'm fine now.", "answer": "I hear that you're saying you're fine now, but it's completely okay if you're still processing whatever made you cry earlier. I'm here if you want to unpack it."},
        
        # TRICKY: Sarcasm
        {"language": "English", "chat_history": "", "user_query": "Oh great, another bot that's going to fix my life.", "answer": "I definitely can't magically fix everything, but I can offer a safe, non-judgmental space for you to vent or explore your thoughts. How has your day been going?"},
    ]
    # Replicate examples to create a robust dataset
    examples = examples * 4
    return [dspy.Example(
        language=ex["language"], 
        chat_history=ex["chat_history"], 
        user_query=ex["user_query"], 
        answer=ex["answer"]
    ).with_inputs("language", "chat_history", "user_query") for ex in examples]


def get_intent_dataset():
    """Dataset for IntentClassifierModule."""
    raw_examples = [
        # Standard Out of Scope
        {"text": "what's the weather like in New York?", "type": "out_of_scope"},
        {"text": "How do I make a bomb?", "type": "out_of_scope"}, 
        {"text": "What is the chemical formula for Serotonin?", "type": "out_of_scope"}, 
        {"text": "Who is Sigmund Freud?", "type": "out_of_scope"}, 
        {"text": "How to cure cancer?", "type": "out_of_scope"}, 
        {"text": "كم سعر البيتكوين اليوم؟", "type": "out_of_scope"},
        {"text": "cómo se hace una paella?", "type": "out_of_scope"},
        {"text": "write a python script to sort an array", "type": "out_of_scope"},
        {"text": "meu computador quebrou, como consertar?", "type": "out_of_scope"},
        {"text": "My boss yelled at me so I need to hire a lawyer.", "type": "out_of_scope"},
        
        # General (Greetings, Goodbyes, Gratitude)
        {"text": "hello, how are you", "type": "greeting"},
        {"text": "مرحبا، كيف حالك؟", "type": "greeting"},
        {"text": "hola, qué tal?", "type": "greeting"},
        {"text": "ہیلو، آپ کیسے ہیں؟", "type": "greeting"},
        {"text": "bonjour, comment ça va?", "type": "greeting"},
        {"text": "see you later", "type": "goodbye"},
        {"text": "مع السلامة", "type": "goodbye"},
        {"text": "adiós, cuídate", "type": "goodbye"},
        {"text": "خدا حافظ", "type": "goodbye"},
        {"text": "au revoir", "type": "goodbye"},
        {"text": "thank you so much", "type": "gratitude"},
        {"text": "شكرا جزيلا لك", "type": "gratitude"},
        {"text": "muchas gracias por tu ayuda", "type": "gratitude"},
        {"text": "بہت شکریہ", "type": "gratitude"},
        {"text": "thx sm", "type": "gratitude"},
        {"text": "Are you a real doctor?", "type": "greeting"}, 
        {"text": "This app is stupid.", "type": "greeting"},
        {"text": "Nevermind, forget it.", "type": "goodbye"},
        {"text": "You saved my life, thanks.", "type": "gratitude"},
        
        # Crisis
        {"text": "I don't want to live anymore.", "type": "crisis"},
        {"text": "أريد أن أؤذي نفسي", "type": "crisis"},
        {"text": "quiero suicidarme", "type": "crisis"},
        {"text": "میں خودکشی کرنا چاہتا ہوں", "type": "crisis"},
        {"text": "je veux me tuer", "type": "crisis"},
        {"text": "i wanna kms", "type": "crisis"},
        {"text": "If I fail this exam I'm going to jump off a bridge.", "type": "crisis"},
        {"text": "Je veux mourir.", "type": "crisis"},
        
        # Tricky Mental Health
        {"text": "I want to kill my anxiety, what medication helps?", "type": "asking_mental_health_question"},
        {"text": "My boss is killing me with this workload.", "type": "asking_mental_health_question"}, 
        {"text": "Should I break up with my girlfriend?", "type": "asking_mental_health_question"}, 
        {"text": "I'm prescribing myself xanax.", "type": "asking_mental_health_question"}, 
        {"text": "My cat died and I can't stop crying.", "type": "asking_mental_health_question"}, 
        {"text": "Estoy loco?", "type": "asking_mental_health_question"},
        {"text": "هل أنا مريض نفسي؟", "type": "asking_mental_health_question"},
        {"text": "طريقة عمل البيتزا", "type": "out_of_scope"},
    ]
    
    # Repeat manual tricky cases so they aren't drowned out by real data
    raw_examples = raw_examples * 2
    
    # ── Inject Real Data (200 examples) ──
    real_data = get_real_mental_health_data(n_samples=200, random_state=4242)
    for row in real_data:
        query = str(row["Context"])[:250]
        raw_examples.append({
            "text": query, 
            "type": "asking_mental_health_question" # All real Amod data is mental health
        })
        
    examples = []
    for ex in raw_examples:
        examples.append(dspy.Example(text=ex["text"], type=ex["type"]).with_inputs("text"))
    return examples

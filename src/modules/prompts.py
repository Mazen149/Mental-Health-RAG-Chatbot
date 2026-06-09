import dspy

# ==============================================================================
# INTENT CLASSIFIER PROMPTS
# ==============================================================================

class IntentClassifierSignature(dspy.Signature):
    """You are a strict intent classification engine for a mental-health support assistant.
    Classify the user's message into exactly one label: greeting, goodbye, gratitude, out_of_scope, asking_mental_health_question, or crisis.
    
    Guidelines:
    - greeting: for greeting, introduction, small talk (e.g. sharing or asking about names, how are you).
    - goodbye: for farewell and parting words.
    - gratitude: for expressions of thanks or appreciation.
    - out_of_scope: for queries completely unrelated to the assistant, user identity, or mental health (e.g. weather, sports, cooking, news, general facts, coding, math).
    - asking_mental_health_question: for any other mental-health-related question, emotional distress, therapy, anxiety, depression, panic, stress, or loneliness.
    - crisis: for any query indicating suicidal thoughts, self-harm, cutting, ending one's life, or intent to inflict harm on oneself.
    """

    text = dspy.InputField(desc="The user message to classify")
    type = dspy.OutputField(desc="exactly one of: greeting, goodbye, gratitude, out_of_scope, asking_mental_health_question, or crisis")
    confidence = dspy.OutputField(desc="confidence score of the intent classification, between 0.0 and 1.0")


class IntentClassifierModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(IntentClassifierSignature)

    def forward(self, text: str) -> dict:
        pred = self.classify(text=text)
        
        pred_type = str(pred.type).strip().strip('"').strip("'").lower()
        valid_types = {
            "greeting", "goodbye", "gratitude", 
            "out_of_scope", "asking_mental_health_question", "crisis"
        }
        if pred_type not in valid_types:
            pred_type = "greeting"
            
        try:
            conf_str = "".join(c for c in str(pred.confidence) if c.isdigit() or c == '.')
            pred_conf = float(conf_str)
        except (ValueError, TypeError):
            pred_conf = 0.65
            
        pred_conf = max(0.0, min(1.0, pred_conf))
        
        return {
            "type": pred_type,
            "confidence": pred_conf,
            "classifier": "llm"
        }

# ==============================================================================
# RAG PIPELINE PROMPTS
# ==============================================================================

class RetrievalRouterSignature(dspy.Signature):
    """Classify if the user's latest query refers to previous chat history (e.g., asking about what they said, their name, their personal details, or what was discussed) and can be answered using only the chat history, or if it asks a new mental health/counseling question that requires retrieving external medical/support document resources."""
    
    chat_history = dspy.InputField(desc="Pruned recent chat turns between user and assistant")
    user_query = dspy.InputField(desc="The user's latest query")
    
    route = dspy.OutputField(desc="exactly 'history_only' or 'requires_retrieval'")


class RetrievalRouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route_predict = dspy.Predict(RetrievalRouterSignature)

    def forward(self, chat_history: str, user_query: str) -> str:
        res = self.route_predict(chat_history=chat_history, user_query=user_query)
        route_val = str(res.route).strip().lower()
        if "history_only" in route_val:
            return "history_only"
        return "requires_retrieval"


class QueryCondenserSignature(dspy.Signature):
    """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    The standalone question MUST be written in English. Do NOT answer the question, just reformulate it
    and output ONLY the standalone question."""
    
    chat_history = dspy.InputField(desc="Recent chat turns between user and assistant")
    user_query = dspy.InputField(desc="The latest user question/query")
    condensed_query = dspy.OutputField(desc="A standalone question in English representing the query in history context")


class QueryCondenserModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.condense = dspy.Predict(QueryCondenserSignature)

    def forward(self, chat_history: str, user_query: str) -> str:
        res = self.condense(chat_history=chat_history, user_query=user_query)
        return str(res.condensed_query).strip()


class GroundedResponseSignature(dspy.Signature):
    """You are a compassionate, professional mental health support assistant.
    Your goal is to provide a supportive, empathetic, and conversational response to the user's query.
    
    CRITICAL GROUNDING RULES:
    - Ground your response and advice in the retrieved contexts. You should also refer to personal facts, context, or situations mentioned in the chat history.
    - If neither the retrieved contexts nor the chat history contain enough information to address the query, respond exactly with: 'I'm sorry, I don't have enough information to answer that.' (or its translation in the user's language).
    - Strictly avoid inventing therapy techniques, clinical diagnoses, or medication names. Do not hallucinate.
    
    CRITICAL LANGUAGE RULES:
    - Respond in the language that the query is ACTUALLY written in.
    - If the contexts are in a different language, translate the relevant information from those contexts into the user's language.
    
    RESPONSE FORMAT:
    - Keep your response concise, to exactly 3-5 sentences.
    - Do not use bullet points or numbered lists.
    - Cite the retrieved contexts by appending the context number in square brackets, e.g. [1], [2], or [3] (corresponding to Context [1], Context [2], or Context [3]). Do not create other citations.
    - Adjust tone implicitly according to the user's emotions and directives. Do not label their emotions explicitly.
    """

    contexts = dspy.InputField(desc="Retrieved counseling case contexts, formatted as Context [1], Context [2], Context [3]")
    emotions = dspy.InputField(desc="User's detected emotional state and tone directives")
    language = dspy.InputField(desc="Detected language and translation/actual query language instructions")
    chat_history = dspy.InputField(desc="Recent chat turns between user and assistant")
    user_query = dspy.InputField(desc="The user's query/message")
    
    answer = dspy.OutputField(desc="Empathetic, grounded response matching language/tone directives and citations")


class GroundedResponseModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GroundedResponseSignature)

    def forward(self, contexts: str, emotions: str, language: str, chat_history: str, user_query: str) -> str:
        res = self.generate(
            contexts=contexts,
            emotions=emotions,
            language=language,
            chat_history=chat_history,
            user_query=user_query
        )
        return str(res.answer).strip()


class GeneralConversationSignature(dspy.Signature):
    """You are Serene AI, a compassionate, friendly, and professional mental health support assistant.
    The user is engaging in greeting, goodbye, gratitude, or basic small talk/introductions.
    Respond warmly, naturally, and in a friendly conversational manner in the user's language.
    If the user has introduced themselves or mentioned their name in the history or query, acknowledge and remember it.
    If they ask for their name, tell them their name if it was mentioned.
    Keep your response concise, to exactly 1-3 sentences. Do not offer clinical advice here; just be warm, welcoming, and supportive.
    """
    
    language = dspy.InputField(desc="The user's language")
    chat_history = dspy.InputField(desc="Recent chat turns between user and assistant")
    user_query = dspy.InputField(desc="The user's query/greeting")
    
    answer = dspy.OutputField(desc="Warm, friendly conversational response in the user's language (1-3 sentences)")


class GeneralConversationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(GeneralConversationSignature)

    def forward(self, language: str, chat_history: str, user_query: str) -> str:
        res = self.respond(language=language, chat_history=chat_history, user_query=user_query)
        return str(res.answer).strip()

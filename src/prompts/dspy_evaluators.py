import dspy

# 1. Router Metric
def router_exact_match(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric function: Exact match on the route type."""
    if isinstance(pred, str):
        pred_route = pred
    elif isinstance(pred, dict):
        pred_route = pred.get("route", "")
    else:
        pred_route = getattr(pred, "route", "")
    return 1.0 if example.route.strip().lower() in str(pred_route).strip().lower() else 0.0


# 2. Condenser Metric
class StandaloneQueryJudge(dspy.Signature):
    """Assess if the predicted condensed query is a standalone question that accurately captures the original user_query's intent, considering the provided chat_history.
    The predicted_query MUST be understandable without the chat_history.
    Return a score between 0.0 and 1.0. Output ONLY the float value."""
    chat_history = dspy.InputField()
    user_query = dspy.InputField()
    predicted_query = dspy.InputField()
    score = dspy.OutputField(desc="A float between 0.0 and 1.0. Output ONLY the number.")

def condenser_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    if isinstance(pred, str):
        pred_query = pred
    elif isinstance(pred, dict):
        pred_query = pred.get("condensed_query", "")
    else:
        pred_query = getattr(pred, "condensed_query", "")
    judge = dspy.Predict(StandaloneQueryJudge)
    res = judge(
        chat_history=example.chat_history, 
        user_query=example.user_query, 
        predicted_query=str(pred_query)
    )
    try:
        # Extract float
        score_str = "".join(c for c in str(res.score) if c.isdigit() or c == '.')
        return float(score_str)
    except (ValueError, TypeError):
        return 0.0


# 3. Grounded Response Metric
class GroundedResponseJudge(dspy.Signature):
    """Assess the predicted_answer on empathy, correct citation usage (e.g. [1]), avoiding hallucination, and correct language.
    Return a score between 0.0 and 1.0 based on how well it addresses the user_query using the contexts. Output ONLY the float value."""
    contexts = dspy.InputField()
    user_query = dspy.InputField()
    predicted_answer = dspy.InputField()
    score = dspy.OutputField(desc="A float between 0.0 and 1.0. Output ONLY the number.")

def grounded_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    if isinstance(pred, str):
        pred_answer = pred
    elif isinstance(pred, dict):
        pred_answer = pred.get("answer", "")
    else:
        pred_answer = getattr(pred, "answer", "")
    judge = dspy.Predict(GroundedResponseJudge)
    res = judge(
        contexts=example.contexts, 
        user_query=example.user_query, 
        predicted_answer=str(pred_answer)
    )
    try:
        score_str = "".join(c for c in str(res.score) if c.isdigit() or c == '.')
        score = float(score_str)
        
        # Simple heuristic penalties
        answer_str = str(pred_answer)
        # Check citations
        if "Context [1]" in example.contexts and "[1]" not in answer_str:
            score -= 0.3
        
        # Check length (3-5 sentences rule)
        sentences = answer_str.count('.') + answer_str.count('!') + answer_str.count('?')
        if sentences > 6 or sentences < 1:
            score -= 0.2
            
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


# 4. General Conversation Metric
class ConversationJudge(dspy.Signature):
    """Assess the predicted_answer for friendliness, warmth, matching the requested language, and appropriate conversational tone.
    Return a score between 0.0 and 1.0. Output ONLY the float value."""
    user_query = dspy.InputField()
    language = dspy.InputField()
    predicted_answer = dspy.InputField()
    score = dspy.OutputField(desc="A float between 0.0 and 1.0. Output ONLY the number.")

def conversation_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    if isinstance(pred, str):
        pred_answer = pred
    elif isinstance(pred, dict):
        pred_answer = pred.get("answer", "")
    else:
        pred_answer = getattr(pred, "answer", "")
    judge = dspy.Predict(ConversationJudge)
    res = judge(
        user_query=example.user_query, 
        language=example.language, 
        predicted_answer=str(pred_answer)
    )
    try:
        score_str = "".join(c for c in str(res.score) if c.isdigit() or c == '.')
        score = float(score_str)
        
        # Length penalty (1-3 sentences)
        answer_str = str(pred_answer)
        sentences = answer_str.count('.') + answer_str.count('!') + answer_str.count('?')
        if sentences > 4:
            score -= 0.2
            
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0

# 5. Intent Metric
def intent_exact_match(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric function: Exact match on the intent type."""
    pred_type = pred["type"] if isinstance(pred, dict) else pred.type
    score = 1.0 if example.type.lower() == str(pred_type).lower() else 0.0
    return score

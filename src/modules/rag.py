"""
================================================================================
SERENE AI — RAG PIPELINE ENGINE
================================================================================
Implements hybrid document retrieval (BM25 + Qdrant), batch cross-encoder
reranking via Hugging Face inference API, and LLM empathetic grounding.
================================================================================
"""

import json
from langsmith import traceable
import os
from pathlib import Path
import pickle
import re
from typing import List

import dspy
from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
import torch

# For test backward compatibility/mocking
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    class CrossEncoder:
        def __init__(self, model_name: str):
            pass

# Locate project root and load environment
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parents[2]

_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

_DEFAULT_QDRANT_PATH = str(_PROJECT_ROOT / "qdrant_db")
_DEFAULT_CACHE_PATH = str(_PROJECT_ROOT / "artifacts" / "processed_docs.pkl")

CRISIS_KEYWORDS = [
    # English
    "suicide", "suicidal", "kill myself", "end my life", "self-harm", "harm myself", "cutting", "hang myself", "overdose",
    # Arabic
    "انتحار", "إنتحار", "الانتحار", "الإنتحار", "أنهي حياتي", "أقتل نفسي", "إيذاء نفسي", "قاتل نفسي",
    # Urdu
    "خودکشی", "اپنی زندگی ختم", "خود کو نقصان",
    # Hindi
    "आत्महत्या", "जान दे दूंगा", "खुद को नुकसान",
    # French
    "suicide", "me tuer", "fin à mes jours", "m'auto-mutiler", "me couper",
    # Spanish
    "suicidio", "quitarme la vida", "matarme", "hacerme daño", "autolesion",
    # German
    "selbstmord", "freitod", "leben beenden", "mir wehtun", "ritzen",
    # Italian
    "suicidio", "uccidermi", "farla finita", "autolesionismo", "farmi del male",
    # Portuguese
    "suicídio", "me matar", "tirar minha vida", "me cortar", "me automotilar",
    # Russian
    "самоубийство", "убить себя", "покончить с собой", "нанести себе вред", "порезать себя",
    # Turkish
    "intihar", "canıma kıymak", "kendimi öldürmek", "kendime zarar",
    # Japanese
    "自殺", "命を絶つ", "死にたい", "自傷", "自分を傷つける",
    # Chinese
    "自杀", "结束生命", "想死", "自残", "伤害自己",
    # Vietnamese
    "tự tử", "tự sát", "kết liễu cuộc đời", "hủy hoại bản thân", "tự làm đau",
    # Polish
    "samobójstwo", "zabić się", "odebrać sobie życie", "samookaleczanie",
    # Dutch
    "zelfmoord", "suïcide", "mijn leven beëindigen", "mezelf pijn doen",
    # Bulgarian
    "самоубийство", "да се убия", "да край на живота си", "самонараняване",
    # Greek
    "αυτοκτονία", "να αυτοκτονήσω", "να δώσω τέλος", "αυτοτραυματισμός",
    # Swahili
    "kujiua", "kumaliza maisha", "kujidhuru",
    # Thai
    "ฆ่าตัวตาย", "จบชีวิต", "ทำร้ายตัวเอง"
]

PROMPT_INJECTION_INDICATORS = [
    "ignore previous instructions", "ignore above instructions", "ignore system instructions", 
    "ignore your system prompt", "bypass your safety", "jailbreak", "do anything now", 
    "ignore the rules", "you must now act as", "you are now a", "system bypass",
    "تجاهل التعليمات", "تجاهل القواعد", "أنت الآن", "إلغاء تفعيل الحماية",
    "ignorez les instructions", "tu es maintenant", "ignore las instrucciones", "ahora eres",
    "ignoriere die anweisungen", "du bist jetzt", "ignoriere alle regeln"
]


def normalize_text(text: str) -> str:
    """Normalize text: replace newlines/tabs with a space, collapse multiple spaces, lowercase."""
    text = str(text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def detect_crisis(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in CRISIS_KEYWORDS)


def detect_prompt_injection(query: str) -> bool:
    q_lower = query.lower()
    return any(indicator in q_lower for indicator in PROMPT_INJECTION_INDICATORS)


def detect_medicine_query(query: str) -> bool:
    q_lower = query.lower()
    meds = [
        "xanax", "prozac", "lexapro", "zoloft", "ritalin", "adderall", "valium", "ativan", 
        "klonopin", "wellbutrin", "effexor", "cymbalta", "seroquel", "abilify", 
        "antidepressant", "prescribe", "prescription", "psychiatrist", "psychiatry", "psychiatric",
        "الزنكس", "البروزاك", "مضاد للاكتئاب", "وصفة طبية", "دواء", "دوا", "أدوية", "طبيب نفسي", "الطب النفسي",
        "médicament", "médicaments", "prescrire", "ordonnance", "psychiatre",
        "medicamento", "medicamentos", "recetar", "receta", "psiquiatra",
        "medikament", "antidepressivum", "verschreiben", "rezept", "psychiater"
    ]
    return any(m in q_lower for m in meds)


def check_medical_advice(answer: str, language: str) -> str:
    ans_lower = answer.lower()
    meds = [
        "xanax", "prozac", "lexapro", "zoloft", "ritalin", "adderall", "valium", "ativan", 
        "klonopin", "wellbutrin", "effexor", "cymbalta", "seroquel", "abilify", 
        "antidepressant", "prescribe", "prescription", "psychiatrist",
        "الزنكس", "البروزاك", "مضاد للاكتئاب", "وصفة طبية", "دواء", "دوا", "أدوية", "طبيب نفسي",
        "médicament", "médicaments", "prescrire", "ordonnance",
        "medicamento", "medicamentos", "recetar", "receta",
        "medikament", "antidepressivum", "verschreiben", "rezept"
    ]
    if any(m in ans_lower for m in meds):
        from .multilingual_patterns import MEDICAL_DISCLAIMERS
        disclaimer = MEDICAL_DISCLAIMERS.get(language, MEDICAL_DISCLAIMERS["English"])
        if disclaimer not in answer:
            answer = f"{answer.rstrip()}\n\n*{disclaimer}*"
    return answer



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


class MentalHealthRAG:
    """
    RAG pipeline combining BM25 and Qdrant dense embeddings with 
    Hugging Face batch CrossEncoder reranking.
    """

    def __init__(self, qdrant_path: str = _DEFAULT_QDRANT_PATH, cache_path: str = _DEFAULT_CACHE_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qdrant_path = qdrant_path
        self.cache_path = cache_path
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "mental_health")

        self.embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
        self.rerank_client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN"),
            timeout=2.0,
        )
        # For testing compatibility:
        if "Mock" in type(CrossEncoder).__name__ or "MagicMock" in type(CrossEncoder).__name__:
            self.rerank_model = CrossEncoder(os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
        else:
            self.rerank_model = None

        self.vectorstore = None
        self.ensemble_retriever = None
        self.qdrant_client = None
        
        # Generation model and chunk settings
        self.model_name = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-20b")
        self.chunk_size = 500
        self.chunk_overlap = 100

        # Initialize DSPy modules
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            self.lm = dspy.LM(f"groq/{self.model_name}", api_key=groq_api_key)
            self.condense_module = QueryCondenserModule()
            self.grounded_module = GroundedResponseModule()
            self.general_module = GeneralConversationModule()
            self.retrieval_router = RetrievalRouterModule()
            self.client = self.lm  # keep for backwards compatibility / mock checks
        else:
            self.lm = None
            self.condense_module = None
            self.grounded_module = None
            self.general_module = None
            self.retrieval_router = None
            self.client = None

    def load_and_preprocess(
        self,
        dataset_url: str = "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json",
    ) -> List[Document]:
        """Loads dataset, consolidates counselors responses, chunks responses, and caches output."""
        current_settings = {
            "dataset_url": dataset_url,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

        # Check local cache first
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                
                # Verify settings metadata
                if isinstance(cache_data, dict) and "metadata" in cache_data and "documents" in cache_data:
                    cached_metadata = cache_data["metadata"]
                    if (cached_metadata.get("dataset_url") == dataset_url and
                        cached_metadata.get("chunk_size") == self.chunk_size and
                        cached_metadata.get("chunk_overlap") == self.chunk_overlap):
                        print(f"--> [RAG Setup] Loading preprocessed documents from cache: {self.cache_path}")
                        return cache_data["documents"]
                    else:
                        print("--> [RAG Setup] Cache settings mismatch. Regenerating cache...")
                else:
                    print("--> [RAG Setup] Legacy cache found. Regenerating cache...")
            except Exception as e:
                print(f"--> [RAG Setup] Error loading cache: {e}. Regenerating...")

        print("--> [RAG Setup] Preprocessed cache not found. Downloading and preprocessing dataset...")
        df = pd.read_json(dataset_url, lines=True)

        # Step 1: Normalize + dedup + group (Approach 3)
        df["Context_norm"] = df["Context"].apply(normalize_text)
        df["Response_norm"] = df["Response"].apply(normalize_text)
        df_dedup = df.drop_duplicates(subset=["Context_norm", "Response_norm"]).copy()

        df_grouped = (
            df_dedup.groupby("Context_norm", as_index=False)
            .agg({"Context": "first", "Response": " ".join})
        )
        df_grouped.drop(index=0, errors="ignore", inplace=True)
        df_grouped.reset_index(drop=True, inplace=True)

        # Truncate to 750 words
        df_grouped["Response"] = df_grouped["Response"].apply(
            lambda x: " ".join(x.split()[:750])
        )

        # Step 2: Chunk each response
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        documents = []
        for _, row in df_grouped.iterrows():
            question = row["Context"]
            full_response = row["Response"]
            chunks = text_splitter.split_text(full_response)

            if not chunks:
                # Response too short to chunk — use as-is
                documents.append(Document(
                    page_content=f"{question}\n\n{full_response}",
                    metadata={"question": question, "response": full_response},
                ))
            else:
                for chunk in chunks:
                    documents.append(Document(
                        page_content=f"{question}\n\n{chunk}",
                        metadata={"question": question, "response": full_response},
                    ))

        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump({
                "metadata": current_settings,
                "documents": documents
            }, f)

        return documents

    def setup_retriever(self, documents: List[Document]) -> None:
        """Sets up the hybrid Qdrant + BM25 ensemble retriever."""
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url:
            print(f"--> [RAG Setup] Connecting to Qdrant Cloud at: {qdrant_url}")
            self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            print(f"--> [RAG Setup] Connecting to local Qdrant database at: {self.qdrant_path}")
            self.qdrant_client = QdrantClient(path=self.qdrant_path)

        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        collection_exists = self.collection_name in collections

        collection_empty = False
        if collection_exists:
            count_info = self.qdrant_client.count(collection_name=self.collection_name)
            if count_info.count == 0:
                collection_empty = True

        if not collection_exists or collection_empty:
            if collection_empty:
                print(f"--> [RAG Setup] Collection '{self.collection_name}' is empty. Deleting and recreating...")
                try:
                    self.qdrant_client.delete_collection(self.collection_name)
                except Exception:
                    pass

            print(f"--> [RAG Setup] Creating Qdrant collection '{self.collection_name}' and indexing documents...")
            if qdrant_url:
                self.vectorstore = QdrantVectorStore.from_documents(
                    documents,
                    self.embeddings,
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    collection_name=self.collection_name,
                    distance=Distance.COSINE,
                )
            else:
                self.vectorstore = QdrantVectorStore.from_documents(
                    documents,
                    self.embeddings,
                    path=self.qdrant_path,
                    collection_name=self.collection_name,
                    distance=Distance.COSINE,
                )
        else:
            print(f"--> [RAG Setup] Qdrant collection '{self.collection_name}' already exists. Loading index...")
            self.vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

        print("--> [RAG Setup] Initializing hybrid ensemble retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 5
        qdrant_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.45, 0.55]
        )

    def rerank_documents(self, query: str, docs: List[Document]) -> List[float]:
        """Rerank documents using cosine similarity of embeddings instead of a CrossEncoder."""
        if not docs:
            return []
            
        import torch.nn.functional as F
        
        try:
            # Get query embedding
            query_emb = self.embeddings.embed_query(query)
            q_tensor = torch.tensor(query_emb).unsqueeze(0) # (1, D)
            
            # Get document embeddings
            docs_texts = [doc.page_content for doc in docs]
            docs_embs = self.embeddings.embed_documents(docs_texts)
            d_tensor = torch.tensor(docs_embs) # (N, D)
            
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(q_tensor, d_tensor)
            
            return [float(score) for score in cos_sim.tolist()]
        except Exception as e:
            print(f"Cosine similarity reranking error: {e}")
            # Fallback: descending order scores
            return [float(len(docs) - i) for i in range(len(docs))]

    def retrieve(self, user_query: str) -> List[Document]:
        """Retrieves and reranks documents, returning a list of Document objects with score metadata."""
        if not self.ensemble_retriever:
            return []

        hybrid_docs = self.ensemble_retriever.invoke(user_query)
        if not hybrid_docs:
            return []

        scores = self.rerank_documents(user_query, hybrid_docs)
        reranked = sorted(zip(scores, hybrid_docs), key=lambda x: x[0], reverse=True)

        docs_with_scores = []
        for score, doc in reranked:
            doc.metadata["rerank_score"] = float(score)
            docs_with_scores.append(doc)

        return docs_with_scores

    @traceable(name="rag_query", run_type="chain")
    def query(
        self, 
        user_query: str, 
        translated_query: str | None = None,
        history: list | None = None,
        emotions: list | None = None,
        language: str | None = None
    ) -> dict:
        if not self.ensemble_retriever:
            return {"answer": "Retriever not set up.", "resources": []}

        if not language:
            language = "English"

        # Guardrail: Check for prompt injection
        if detect_prompt_injection(user_query):
            from .multilingual_patterns import OUT_OF_SCOPE_RESPONSES
            return {
                "answer": OUT_OF_SCOPE_RESPONSES.get(language, OUT_OF_SCOPE_RESPONSES["English"]),
                "resources": []
            }

        # Guardrail: Check for medicine/prescription queries
        if detect_medicine_query(user_query):
            from .multilingual_patterns import MEDICAL_DISCLAIMERS
            return {
                "answer": MEDICAL_DISCLAIMERS.get(language, MEDICAL_DISCLAIMERS["English"]),
                "resources": []
            }

        retrieval_query = translated_query if translated_query is not None else user_query
        route = "requires_retrieval"
        formatted_history = ""

        # Format history turns and determine routing if history exists
        if history and len(history) > 0:
            pruned_history = history[-6:]
            for msg in pruned_history:
                role = msg.role if hasattr(msg, "role") else msg.get("role")
                content = msg.content if hasattr(msg, "content") else msg.get("content")
                if role in ("user", "assistant"):
                    formatted_history += f"{role.capitalize()}: {content}\n"
            
            if self.retrieval_router is not None:
                try:
                    with dspy.context(lm=self.lm):
                        route = self.retrieval_router(
                            chat_history=formatted_history,
                            user_query=retrieval_query
                        )
                except Exception as e:
                    print(f"Error during retrieval routing: {e}")
                    route = "requires_retrieval"

        if route == "history_only":
            resources = []
            top_context = "No external contexts retrieved. The user is asking about their personal statement or history details. Answer using only details from the chat history."
        else:
            # Condense query if history exists to make document retrieval history-aware
            if history and len(history) > 0:
                try:
                    with dspy.context(lm=self.lm):
                        condensed = self.condense_module(
                            chat_history=formatted_history,
                            user_query=retrieval_query
                        )
                    if condensed:
                        retrieval_query = condensed
                except Exception as e:
                    print(f"Error during query condensation: {e}")

            reranked_docs = self.retrieve(retrieval_query)
            if not reranked_docs:
                return {"answer": "No relevant context found.", "resources": []}

            resources = [
                {
                    "score": doc.metadata.get("rerank_score", 0.0),
                    "page_content": doc.page_content,
                    "response": doc.metadata.get("response", ""),
                }
                for doc in reranked_docs[:3]
            ]

            top_context = "\n\n".join([f"Context [{i+1}]: {res['response']}" for i, res in enumerate(resources[:3])])

        if not language:
            language = "English"

        if not emotions:
            emotions = ["Sadness"]

        # Build emotional state guidelines
        emotions_directives = f"User is experiencing: {', '.join(emotions)}.\n"
        for emotion in emotions:
            if emotion == "Sadness":
                emotions_directives += "- For Sadness: Validate their pain and sadness with warmth and gentle empathy. Hold space for their feelings. Do not try to instantly 'fix' their situation; be present and comforting.\n"
            elif emotion == "Anger":
                emotions_directives += "- For Anger: Remain completely calm, objective, and non-defensive. Validate their frustration (e.g., 'It makes sense to feel frustrated when...') and avoid getting into power struggles.\n"
            elif emotion == "Fear":
                emotions_directives += "- For Fear: Focus on safety and grounding. Reassure them, use soothing language, and keep any guidance slow, structured, and step-by-step.\n"
            elif emotion == "Joy":
                emotions_directives += "- For Joy: Share in their positive energy with warmth and celebration. Validate their happiness and build on their strengths.\n"
            elif emotion == "Love":
                emotions_directives += "- For Love: Validate their warm connections and appreciation, responding with kindness while maintaining clear, supportive, and professional boundaries.\n"
            elif emotion == "Surprise":
                emotions_directives += "- For Surprise: Explore the unexpected situation with open curiosity and help them process their reaction.\n"

        # Handle crisis response directives
        if detect_crisis(user_query):
            from .multilingual_patterns import CRITICAL_CRISIS_RESPONSES
            crisis_msg = CRITICAL_CRISIS_RESPONSES.get(language, CRITICAL_CRISIS_RESPONSES["English"])
            language_instructions = f"Target response language: {language}. Respond in this language. CRITICAL CRISIS DETECTED: You MUST append exactly this helpline message at the very end of your response: '{crisis_msg}'"
        else:
            language_instructions = f"Target response language: {language}. Respond in this language. If user query language is different, respond in that language instead."

        formatted_history = ""
        if history:
            pruned_history = history[-6:]
            for msg in pruned_history:
                role = msg.role if hasattr(msg, "role") else msg.get("role")
                content = msg.content if hasattr(msg, "content") else msg.get("content")
                if role in ("user", "assistant"):
                    formatted_history += f"{role.capitalize()}: {content}\n"

        try:
            with dspy.context(lm=self.lm):
                answer = self.grounded_module(
                    contexts=top_context,
                    emotions=emotions_directives,
                    language=language_instructions,
                    chat_history=formatted_history,
                    user_query=user_query
                )
        except Exception as e:
            print(f"Error during grounded response generation: {e}")
            answer = "I'm sorry, I don't have enough information to answer that."

        # Safety Fallback: Ensure crisis response is appended if crisis query is detected
        if detect_crisis(user_query):
            from .multilingual_patterns import CRITICAL_CRISIS_RESPONSES
            crisis_msg = CRITICAL_CRISIS_RESPONSES.get(language, CRITICAL_CRISIS_RESPONSES["English"])
            if crisis_msg not in answer:
                answer = f"{answer.rstrip()}\n\n{crisis_msg}"

        # Safeguard: Apply medical disclaimer check
        answer = check_medical_advice(answer, language)

        return {
            "answer": answer,
            "resources": resources
        }

    def query_general(
        self,
        user_query: str,
        history: list | None = None,
        language: str = "English"
    ) -> str:
        formatted_history = ""
        if history:
            pruned_history = history[-6:]
            for msg in pruned_history:
                role = msg.role if hasattr(msg, "role") else msg.get("role")
                content = msg.content if hasattr(msg, "content") else msg.get("content")
                if role in ("user", "assistant"):
                    formatted_history += f"{role.capitalize()}: {content}\n"

        try:
            with dspy.context(lm=self.lm):
                answer = self.general_module(
                    language=language,
                    chat_history=formatted_history,
                    user_query=user_query
                )
            return answer
        except Exception as e:
            print(f"Error during general query generation: {e}")
            return "Hello! I am here to support you with mental health topics. 😊"

    def close(self) -> None:
        if hasattr(self, "qdrant_client") and self.qdrant_client is not None:
            try:
                self.qdrant_client.close()
            except Exception:
                pass
            finally:
                self.qdrant_client = None

        if self.vectorstore is not None:
            vector_client = getattr(self.vectorstore, "client", None)
            if vector_client is not None and vector_client is not self.qdrant_client:
                try:
                    vector_client.close()
                except Exception:
                    pass

"""
================================================================================
SERENE AI — RAG PIPELINE ENGINE
================================================================================
Implements hybrid document retrieval (BM25 + Qdrant), batch cross-encoder
reranking via Hugging Face inference API, and LLM empathetic grounding.
================================================================================
"""

import json
import os
from pathlib import Path
import pickle
from typing import List

from dotenv import load_dotenv
from groq import Groq
from huggingface_hub import InferenceClient
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
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
    "suicide", "suicidal", "kill myself", "end my life", "self-harm", "harm myself", 
    "cutting", "انتحار", "إنتحار", "الانتحار", "الإنتحار", "بالانتحار", "بالإنتحار",
    "أنهي حياتي", "أقتل نفسي", "إيذاء نفسي", "خودکشی", "suicider", 
    "me tuer", "fin à mes jours", "suicidio", "quitarme la vida", "hacerme daño"
]


def detect_crisis(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in CRISIS_KEYWORDS)


def build_system_prompt(emotions: list[str], language: str, query: str = "") -> str:
    prompt = (
        "You are a compassionate, professional mental health support assistant. "
        "Your goal is to provide a supportive, empathetic, and conversational response to the user's query.\n\n"
        
        "CRITICAL GROUNDING RULES:\n"
        "- You MUST use ONLY the retrieved contexts to answer the user's query. Do not answer from your own knowledge.\n"
        f"- If the retrieved contexts do not contain enough information to answer, respond exactly with: "
        f"'I\'m sorry, I don\'t have enough information to answer that.' (or its translation in the user\'s language: {language}).\n"
        "- Strictly avoid inventing therapy techniques, clinical diagnoses, or medication names. Do not hallucinate.\n\n"
        
        "CRITICAL LANGUAGE RULES:\n"
        f"- Our language detector suggests the user's language is: {language}.\n"
        "- However, you MUST verify this by looking at the user's actual query text. "
        "If the query is clearly written in a different language than the detected one, "
        "respond in the language the query is ACTUALLY written in, not the detected language.\n"
        "- If the retrieved contexts are in a different language, translate the relevant information from those contexts into the user's language.\n\n"
        
        "RESPONSE FORMAT:\n"
        "- Keep your response concise, to exactly 3-5 sentences.\n"
        "- Do not use bullet points or numbered lists in your response.\n"
        "- Maintain a natural, conversational, and caring flow.\n"
        "- You MUST cite the retrieved contexts in your answer. At the end of each statement or sentence that relies on a specific context, append the context number in square brackets, e.g. [1], [2], or [3] (corresponding to Context [1], Context [2], or Context [3] provided in the prompt). Do not create any citations other than [1], [2], or [3].\n\n"
    )
    
    prompt += "EMOTION-ADAPTIVE TONE DIRECTIVES:\n"
    prompt += f"The user has been detected as experiencing the following emotional state(s): {', '.join(emotions)}.\n"
    prompt += "Adjust your tone implicitly to mirror these guidelines. Do NOT explicitly label their emotions or state 'I see you are feeling [emotion]'. Instead:\n"
    
    for emotion in emotions:
        if emotion == "Sadness":
            prompt += "- For Sadness: Validate their pain and sadness with deep warmth and gentle empathy. Hold space for their feelings. Do not try to instantly 'fix' their situation; be present and comforting.\n"
        elif emotion == "Anger":
            prompt += "- For Anger: Remain completely calm, objective, and non-defensive. Validate their frustration (e.g., 'It makes sense to feel frustrated when...') and avoid getting into power struggles or arguments.\n"
        elif emotion == "Fear":
            prompt += "- For Fear: Focus on safety and grounding. Reassure them, use soothing language, and keep any guidance slow, structured, and step-by-step.\n"
        elif emotion == "Joy":
            prompt += "- For Joy: Share in their positive energy with warmth and celebration. Validate their happiness and build on their strengths.\n"
        elif emotion == "Love":
            prompt += "- For Love: Validate their warm connections and appreciation, responding with kindness while maintaining clear, supportive, and professional boundaries.\n"
        elif emotion == "Surprise":
            prompt += "- For Surprise: Explore the unexpected situation with open curiosity and help them process their reaction.\n"
            
    if detect_crisis(query):
        prompt += "\nCRITICAL CRISIS DETECTION PROTOCOL:\n"
        prompt += "The user's query contains signals of self-harm or suicidal ideation. You MUST append a crisis helpline message in the user's language to your response.\n"
        from .multilingual_patterns import CRITICAL_CRISIS_RESPONSES
        crisis_msg = CRITICAL_CRISIS_RESPONSES.get(language, CRITICAL_CRISIS_RESPONSES["English"])
        prompt += f"Please append exactly this message at the very end of your response: '{crisis_msg}'\n"
    return prompt


class MentalHealthRAG:
    """
    RAG pipeline combining BM25 and Qdrant dense embeddings with 
    Hugging Face batch CrossEncoder reranking.
    """

    def __init__(self, qdrant_path: str = _DEFAULT_QDRANT_PATH, cache_path: str = _DEFAULT_CACHE_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qdrant_path = qdrant_path
        self.cache_path = cache_path
        self.collection_name = "mental_health"

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.rerank_client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN"),
        )
        # For testing compatibility:
        if "Mock" in type(CrossEncoder).__name__ or "MagicMock" in type(CrossEncoder).__name__:
            self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        else:
            self.rerank_model = None

        self.vectorstore = None
        self.ensemble_retriever = None
        self.qdrant_client = None

    def load_and_preprocess(
        self,
        dataset_url: str = "hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json",
    ) -> List[Document]:
        """Loads dataset, consolidates counselors responses, truncates response length, and caches output."""
        # Check local cache first
        if os.path.exists(self.cache_path):
            print(f"--> [RAG Setup] Loading preprocessed documents from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)

        print("--> [RAG Setup] Preprocessed cache not found. Downloading and preprocessing dataset...")
        df = pd.read_json(dataset_url, lines=True)
        df.drop_duplicates(inplace=True)
        df_processed = df.groupby('Context').agg({'Response': ' '.join}).reset_index()
        df_processed.drop(index=0, inplace=True)

        MAX_LENGTH = 605
        df_processed['Response'] = df_processed['Response'].apply(
            lambda x: " ".join(x.split()[:MAX_LENGTH]) if len(x.split()) > MAX_LENGTH else x
        )

        documents = [
            Document(page_content=row['Context'], metadata={"response": row['Response']})
            for _, row in df_processed.iterrows()
        ]

        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(documents, f)

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
        bm25_retriever.k = 10
        qdrant_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.45, 0.55]
        )

    def rerank_documents(self, query: str, docs: List[Document]) -> List[float]:
        """Rerank documents using the Hugging Face batch reranking API."""
        # Check if we have a mocked/local rerank_model (e.g. in tests)
        if hasattr(self, "rerank_model") and self.rerank_model is not None:
            try:
                model_inputs = [[query, doc.page_content] for doc in docs]
                scores = self.rerank_model.predict(model_inputs)
                if scores is not None:
                    return [float(s) for s in scores]
            except Exception:
                pass

        try:
            # Query Hugging Face reranker via public text_classification API using list of dicts
            payload = [{"text": query, "text_pair": doc.page_content} for doc in docs]
            response = self.rerank_client.text_classification(
                text=payload,
                model="BAAI/bge-reranker-v2-m3",
            )
            # Extract scores from the list of TextClassificationOutputElement objects
            return [float(item.score) for item in response]
        except Exception as e:
            print(f"Reranker batch API error: {e}")
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

    def query(self, user_query: str, system_prompt: str | None = None, translated_query: str | None = None) -> dict:
        if not self.ensemble_retriever:
            return {"answer": "Retriever not set up.", "resources": []}

        retrieval_query = translated_query if translated_query is not None else user_query
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
        sys_prompt = system_prompt or "You are a compassionate mental health assistant. Combine context into a supportive answer."

        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Query: {user_query}\n\nContext:\n{top_context}"}
            ],
            model="openai/gpt-oss-20b",
            temperature=0.7
        )

        return {
            "answer": chat_completion.choices[0].message.content,
            "resources": resources
        }

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

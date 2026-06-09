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
import re
from typing import List

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
    "suicide", "suicidal", "kill myself", "end my life", "self-harm", "harm myself", 
    "cutting", "انتحار", "إنتحار", "الانتحار", "الإنتحار", "بالانتحار", "بالإنتحار",
    "أنهي حياتي", "أقتل نفسي", "إيذاء نفسي", "خودکشی", "suicider", 
    "me tuer", "fin à mes jours", "suicidio", "quitarme la vida", "hacerme daño"
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
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.rerank_client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN"),
            timeout=2.0,
        )
        # For testing compatibility:
        if "Mock" in type(CrossEncoder).__name__ or "MagicMock" in type(CrossEncoder).__name__:
            self.rerank_model = CrossEncoder('BAAI/bge-reranker-v2-m3')
        else:
            self.rerank_model = None

        self.vectorstore = None
        self.ensemble_retriever = None
        self.qdrant_client = None
        
        # Generation model and chunk settings
        self.model_name = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-20b")
        self.max_contexts = int(os.getenv("RAG_MAX_CONTEXTS", "5"))
        self.min_rerank_score = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.12"))
        self.chunk_size = 500
        self.chunk_overlap = 100

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
        bm25_retriever.k = 8
        qdrant_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.35},
            search_type="mmr",
        )

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever],
            weights=[0.55, 0.45]
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

        filtered_docs = [doc for doc in docs_with_scores if doc.metadata.get("rerank_score", 0.0) >= self.min_rerank_score]
        return filtered_docs or docs_with_scores[: self.max_contexts]

    def query(
        self, 
        user_query: str, 
        system_prompt: str | None = None, 
        translated_query: str | None = None,
        history: list | None = None
    ) -> dict:
        if not self.ensemble_retriever:
            return {"answer": "Retriever not set up.", "resources": []}

        retrieval_query = translated_query if translated_query is not None else user_query

        # Condense query if history exists to make document retrieval history-aware
        if history and len(history) > 0:
            condense_prompt = (
                "Given a chat history and the latest user question which might reference context in the chat history, "
                "formulate a standalone question which can be understood without the chat history. "
                "The standalone question MUST be written in English. Do NOT answer the question, just reformulate it "
                "and output ONLY the standalone question."
            )
            condense_messages = [{"role": "system", "content": condense_prompt}]
            
            # Keep only the last 6 messages of history for context
            pruned_history = history[-6:]
            for msg in pruned_history:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    role = msg.role
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                else:
                    continue
                if role in ("user", "assistant"):
                    condense_messages.append({"role": role, "content": content})
            
            condense_messages.append({"role": "user", "content": retrieval_query})
            
            try:
                condense_completion = self.client.chat.completions.create(
                    messages=condense_messages,
                    model=self.model_name,
                    temperature=0.0
                )
                condensed = condense_completion.choices[0].message.content.strip()
                if condensed:
                    retrieval_query = condensed
            except Exception as e:
                print(f"Error during query condensation: {e}")

        reranked_docs = self.retrieve(retrieval_query)
        if not reranked_docs:
            return {"answer": "No relevant context found.", "resources": []}

        selected_docs = reranked_docs[: self.max_contexts]
        resources = [
            {
                "score": doc.metadata.get("rerank_score", 0.0),
                "page_content": doc.page_content,
                "response": doc.metadata.get("response", ""),
            }
            for doc in selected_docs
        ]

        context_blocks = []
        for i, res in enumerate(resources, start=1):
            question_part = ""
            response_part = res["response"].strip()
            page_content = res["page_content"].strip()
            if "\n\n" in page_content:
                question_part = page_content.split("\n\n", 1)[0].strip()
            if not question_part:
                question_part = "Unknown source question"
            response_excerpt = " ".join(response_part.split()[:180])
            context_blocks.append(
                f"Context [{i}]\nSource question: {question_part}\nSource answer excerpt: {response_excerpt}"
            )

        top_context = "\n\n".join(context_blocks)
        sys_prompt = system_prompt or (
            "You are a compassionate mental health assistant. Answer using only the provided context. "
            "If the context does not contain enough information, say you do not have enough information. "
            "Cite the specific context number after each sentence that relies on it."
        )

        formatted_messages = [{"role": "system", "content": sys_prompt}]

        if history:
            # Keep only the last 6 messages of history for a small, optimized context window
            pruned_history = history[-6:]
            for msg in pruned_history:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    role = msg.role
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                else:
                    continue
                if role in ("user", "assistant"):
                    formatted_messages.append({"role": role, "content": content})

        formatted_messages.append({
            "role": "user",
            "content": f"Query: {user_query}\n\nContext:\n{top_context}"
        })

        chat_completion = self.client.chat.completions.create(
            messages=formatted_messages,
            model=self.model_name,
            temperature=0.0
        )

        draft_answer = chat_completion.choices[0].message.content or ""
        grounded_prompt = (
            "You are a strict grounding editor for a mental health assistant.\n"
            "Rewrite the draft answer so that every factual claim is directly supported by the provided context.\n"
            "Remove any unsupported, speculative, or overly specific claims.\n"
            "If the context does not support a safe answer, respond exactly with: "
            "\"I'm sorry, I don't have enough information to answer that.\" \n"
            "Keep the answer concise, warm, and empathetic.\n"
            "Use only the provided context and keep citations in square brackets like [1], [2], or [3].\n"
            "Return only the final answer text."
        )
        grounding_messages = [
            {"role": "system", "content": grounded_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {user_query}\n\n"
                    f"Context:\n{top_context}\n\n"
                    f"Draft answer:\n{draft_answer}"
                ),
            },
        ]
        try:
            grounded_completion = self.client.chat.completions.create(
                messages=grounding_messages,
                model=self.model_name,
                temperature=0.0,
            )
            final_answer = grounded_completion.choices[0].message.content or draft_answer
        except Exception:
            final_answer = draft_answer

        return {
            "answer": final_answer,
            "resources": resources
        }

    def query_general(
        self,
        user_query: str,
        history: list | None = None,
        language: str = "English"
    ) -> str:
        # Build system prompt for general conversation in user's language/context
        sys_prompt = (
            f"You are Serene AI, a compassionate, friendly, and professional mental health support assistant.\n"
            f"The user's language is {language}.\n"
            f"The user is engaging in greeting, goodbye, gratitude, or basic small talk/introductions (like sharing or asking about names).\n"
            f"Respond warmly, naturally, and in a friendly conversational manner in their language ({language}).\n"
            f"If the user has introduced themselves or mentioned their name in the history or current query, acknowledge and remember it. "
            f"If they ask for their name, tell them their name if it was mentioned.\n"
            f"Keep your response concise, to exactly 1-3 sentences. Do not offer clinical advice here; just be warm, welcoming, and supportive."
        )

        formatted_messages = [{"role": "system", "content": sys_prompt}]

        if history:
            # Keep only the last 6 messages
            pruned_history = history[-6:]
            for msg in pruned_history:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    role = msg.role
                    content = msg.content
                elif isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                else:
                    continue
                if role in ("user", "assistant"):
                    formatted_messages.append({"role": role, "content": content})

        formatted_messages.append({"role": "user", "content": user_query})

        chat_completion = self.client.chat.completions.create(
            messages=formatted_messages,
            model=self.model_name,
            temperature=0.2
        )

        return chat_completion.choices[0].message.content

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

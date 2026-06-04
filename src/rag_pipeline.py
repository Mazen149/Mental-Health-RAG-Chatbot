import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from typing import List
from huggingface_hub import InferenceClient

load_dotenv()

SYSTEM_PROMPT = """
You are a compassionate mental health assistant. Use the following retrieved contexts to provide a supportive and informative answer to the user's query.

CRITICAL LANGUAGE RULE: You must detect the language of the user's query and respond in that EXACT same language. If the retrieved contexts are in a different language, you must translate the relevant information from those contexts into the user's language. Do not reply in the language of the context if it differs from the user's query.

Combine information from multiple contexts if necessary, and ensure your response is empathetic and helpful.
Do not answer from your own knowledge - only use the provided contexts. If the contexts do not contain relevant information, respond with an appropriate fallback message (like "I'm sorry, I don't have enough information to answer that.") translated into the user's language.
"""

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "self-harm", "harm myself", 
    "cutting", "انتحار", "أنهي حياتي", "أقتل نفسي", "إيذاء نفسي", "suicider", 
    "me tuer", "fin à mes jours", "suicidio", "quitarme la vida", "hacerme daño"
]

def detect_crisis(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in CRISIS_KEYWORDS)

def build_system_prompt(emotions: list[str], language: str, query: str = "") -> str:
    # 1. Base Prompt with Grounding, Anti-Hallucination, Format
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
        "- Maintain a natural, conversational, and caring flow.\n\n"
    )
    
    # 2. Add Emotion-Adaptive Tone
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
            
    # 3. Add Crisis Detection
    if detect_crisis(query):
        prompt += "\nCRITICAL CRISIS DETECTION PROTOCOL:\n"
        prompt += "The user's query contains signals of self-harm or suicidal ideation. You MUST append a crisis helpline message in the user's language to your response.\n"
        if language == "Arabic":
            prompt += "Please append exactly this message at the very end of your response: 'إذا كنت تمر بأزمة أو تراودك أفكار لإيذاء نفسك، يرجى التواصل للحصول على دعم فوري. يمكنك الاتصال بخط المساعدة الوطني للسلامة النفسية، أو التوجه إلى أقرب مستشفى طوارئ.'\n"
        elif language == "French":
            prompt += "Please append exactly this message at the very end of your response: 'Si vous êtes en détresse ou si vous pensez à vous faire du mal, veuillez contacter immédiatement une ligne d'aide d'urgence ou composer le 3114 (Écoute Suicide).'\n"
        elif language == "Spanish":
            prompt += "Please append exactly this message at the very end of your response: 'Si estás en crisis o tienes pensamientos de hacerte daño, por favor busca apoyo de inmediato. Puedes llamar o enviar un mensaje al 988 para la Línea de Prevención del Suicidio y Crisis.'\n"
        else:
            prompt += "Please append exactly this message at the very end of your response: 'If you are in distress or having thoughts of self-harm, please reach out for immediate support. You can call or text the Suicide & Crisis Lifeline at 988 (available 24/7), or go to your nearest emergency room.'\n"
            
    return prompt


class MentalHealthRAG:
    def __init__(self, qdrant_path=None, cache_path=None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.qdrant_path = os.path.abspath(
            qdrant_path
            if qdrant_path is not None
            else os.path.join(base_dir, "qdrant_db")
        )
        self.cache_path = os.path.abspath(
            cache_path
            if cache_path is not None
            else os.path.join(base_dir, "processed_docs.pkl")
        )

        # FIXED: Removed duplicate bare SentenceTransformer initialization to save memory/time
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # self.rerank_model = CrossEncoder("Alibaba-NLP/gte-multilingual-reranker-base", trust_remote_code=True, device=self.device)
        # self.rerank_model.model = self.rerank_model.model.to(torch.float32)  # Ensure correct dtype
        self.rerank_client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN"),
        )
        self.vectorstore = None
        self.ensemble_retriever = None
        self.collection_name = "mental_health"
        self.qdrant_client = None

    def close(self) -> None:
        """Release local resources explicitly before interpreter shutdown."""
        qdrant_client = getattr(self, "qdrant_client", None)
        if qdrant_client is not None:
            try:
                qdrant_client.close()
            finally:
                self.qdrant_client = None

        if self.vectorstore is not None:
            vector_client = getattr(self.vectorstore, "client", None)
            if vector_client is not None and vector_client is not qdrant_client:
                try:
                    vector_client.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def load_and_preprocess(
        self,
        dataset_url="hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json",
    ) -> list:
        """Loads dataset from local cache if available, otherwise downloads from HF and caches it."""
        # Check if local cache exists
        if os.path.exists(self.cache_path):
            print(
                f"--> Loading preprocessed documents from local cache: {self.cache_path}"
            )
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)

        print(
            "--> Cache not found. Downloading and processing dataset from Hugging Face..."
        )
        df = pd.read_json(dataset_url, lines=True)
        df.drop_duplicates(inplace=True)
        df_processed = df.groupby("Context").agg({"Response": " ".join}).reset_index()
        df_processed.drop(index=0, inplace=True)

        # Kept your response mapping
        documents = [
            Document(
                page_content=row["Context"], metadata={"response": row["Response"]}
            )
            for _, row in df_processed.iterrows()
        ]

        # Save to local cache for next run
        print(f"--> Saving preprocessed documents to cache: {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            pickle.dump(documents, f)

        return documents

    def setup_retriever(self, documents: List[Document]) -> None:
        """Sets up retrievers. Instantly loads Qdrant vectors from disk if they already exist."""

        # Only open the local Qdrant client if the collection already exists.
        collections = []
        if os.path.exists(self.qdrant_path):
            self.qdrant_client = QdrantClient(path=self.qdrant_path)
            collections = [
                c.name for c in self.qdrant_client.get_collections().collections
            ]

        if self.collection_name not in collections:
            if self.qdrant_client is not None:
                self.qdrant_client.close()
                self.qdrant_client = None

            print(
                "--> Qdrant collection not found. Generating dense embeddings (this may take a few minutes)..."
            )
            self.vectorstore = QdrantVectorStore.from_documents(
                documents,
                self.embeddings,
                path=self.qdrant_path,
                collection_name=self.collection_name,
                distance=Distance.COSINE,
            )
        else:
            print(
                "--> Existing Qdrant index found! Loading vectors from disk instantly..."
            )
            self.vectorstore = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

        print("--> Building BM25 index...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10

        qdrant_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, qdrant_retriever], weights=[0.45, 0.55]
        )
        print("--> Hybrid Retriever setup complete.")

    def query(self, user_query: str, system_prompt: str | None = None) -> dict:
        if not self.ensemble_retriever:
            return {"answer": "Retriever not set up.", "resources": []}

        hybrid_docs = self.ensemble_retriever.invoke(user_query)
        if not hybrid_docs:
            return {"answer": "No relevant context found.", "resources": []}

        try:
            # Ensure inputs are lists of strings and have reasonable length
            scores = self.rerank_documents(
                user_query,
                hybrid_docs
            )

            reranked = sorted(
                zip(scores, hybrid_docs),
                key=lambda x: x[0],
                reverse=True
            )

            print(f"DEBUG: Raw scores from reranker: {scores}")
            # reranked = sorted(
            #     zip(scores, hybrid_docs), key=lambda x: x[0], reverse=True
            # )

        except (IndexError, RuntimeError) as e:
            print(f"Warning: Reranking failed ({e}). Using retrieval order instead.")
            # Use reverse indices as scores to preserve ranking
            reranked = [
                (float(len(hybrid_docs) - i), doc) for i, doc in enumerate(hybrid_docs)
            ]

        resources = [
            {
                "score": float(score),
                "page_content": doc.page_content,
                "response": doc.metadata.get("response", ""),
            }
            for score, doc in reranked[:3]
        ]

        top_context = "\n\n".join(
            [f"Context: {resource['response']}" for resource in resources]
        )

        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"Query: {user_query}\n\nContext:\n{top_context}",
                },
            ],
            model="openai/gpt-oss-20b",
            temperature=0.2,
        )
        return {
            "answer": response.choices[0].message.content,
            "resources": resources,
        }
        
    def rerank_documents(self, query: str, docs: List[Document]):
        """Rerank documents using the HuggingFace batch reranking API.
        
        Sends a single batch request with query + all document texts,
        returns a list of relevance scores in the original document order.
        """
        texts = [doc.page_content for doc in docs]
        
        try:
            response = self.rerank_client.post(
                json={"query": query, "texts": texts, "top_n": len(texts)},
                model="BAAI/bge-reranker-v2-m3",
            )
            results = json.loads(response)
            
            # Map scores back to original document order
            scores = [0.0] * len(docs)
            for item in results:
                scores[item["index"]] = item["score"]
            return scores
            
        except Exception as e:
            print(f"Reranker batch API error: {e}")
            # Fallback: preserve retrieval order with descending scores
            return [float(len(docs) - i) for i in range(len(docs))]


if __name__ == "__main__":
    # rag = MentalHealthRAG()
    # try:
    #     # 1. This will check local cache file first
    #     documents = rag.load_and_preprocess()

    #     # 2. This will check Qdrant local database folder first
    #     rag.setup_retriever(documents)

    #     # 3. Execution
    #     test_query = "انا مكتئب جدا، ماذا أفعل؟"
    #     print(f"\nQuery: {test_query}")
    #     result = rag.query(test_query)
    #     print("\nAnswer:", result["answer"])
    #     print("\nResources:", result["resources"])

    #     test_query = "i am depressed, what should i do?"
    #     print(f"\nQuery: {test_query}")
    #     result = rag.query(test_query)
    #     print("\nAnswer:", result["answer"])
    #     print("\nResources:", result["resources"])
    # finally:
    #     rag.close()
    
    result = client.text_classification(
    "How do I reset my password? [SEP] Click forgot password to reset it.",
    model="BAAI/bge-reranker-v2-m3",
)

    print(result)

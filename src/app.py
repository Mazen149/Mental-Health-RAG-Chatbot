"""
================================================================================
SERENE AI — FASTAPI APPLICATION SERVER
================================================================================
Empathetic, multi-layered mental health support backend leveraging
asynchronous model loading, hybrid retrieval (BM25 + Qdrant), and LLM grounding.
================================================================================
"""

import importlib
import os
from pathlib import Path
import sys
from typing import List


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# 1. Environment Loading & Configuration
# ------------------------------------------------------------------------------
from .config import config

# Local project imports (must load after environment resolution)
from .modules.rag import MentalHealthRAG
from .router import route_query

# ------------------------------------------------------------------------------
# 2. Pydantic API Schemas
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message sender: 'user' or 'assistant'."
    )
    content: str = Field(
        ...,
        description="The text content of the message."
    )


class ChatRequest(BaseModel):
    query: str = Field(
        ..., 
        description="The user's query or message to the mental health chatbot."
    )
    history: List[ChatMessage] | None = Field(
        None,
        description="The recent chat history turns for conversational context."
    )


class Resource(BaseModel):
    score: float = Field(
        ..., 
        description="The semantic or reranked relevance score of this document context."
    )
    page_content: str = Field(
        ..., 
        description="The counseling context representing the user's situation."
    )
    response: str = Field(
        ..., 
        description="The verified clinical advice or response corresponding to this context."
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        ..., 
        description="The empathetic, model-generated response grounded in grounding contexts."
    )
    resources: List[Resource] = Field(
        ..., 
        description="The list of grounding counseling cases and clinician advices retrieved."
    )
    language: str | None = Field(
        None, 
        description="The detected language of the user query."
    )
    emotion: List[str] | None = Field(
        None, 
        description="The detected emotional states from emotion classifier."
    )
    intent: str | None = Field(
        None, 
        description="The classified conversational or clinical intent of the query."
    )


# ------------------------------------------------------------------------------
# 3. Environment Validation Helper
# ------------------------------------------------------------------------------
def validate_environment() -> None:
    """
    Validates essential environment variables, model artifacts, required packages,
    and Qdrant connectivity before server boot.
    """
    errors = []
    
    # Verify environment API variables
    groq_api_key = config.GROQ_API_KEY
    hf_token = config.HF_TOKEN
    qdrant_url = config.QDRANT_URL
    qdrant_api_key = config.QDRANT_API_KEY
    
    qdrant_local_path = str(config.QDRANT_LOCAL_PATH)
    
    if not groq_api_key:
        errors.append("Missing environment variable: GROQ_API_KEY")
    if not hf_token:
        errors.append("Missing environment variable: HF_TOKEN")
        
    qdrant_ok = False
    if qdrant_url:
        qdrant_ok = True
    elif os.path.exists(qdrant_local_path):
        qdrant_ok = True
        
    if not qdrant_ok:
        errors.append(
            f"Qdrant database not configured. Neither QDRANT_URL is set, "
            f"nor does local Qdrant database exist at: {qdrant_local_path}"
        )
        
    # Verify mandatory dependencies
    required_packages = [
        "peft", "transformers", "sentence_transformers", 
        "qdrant_client", "langchain_qdrant", "groq", "fastapi"
    ]
    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            errors.append(f"Missing required pip package: {pkg}")
            
    # Verify Module 1 Language Detection model pickles
    mod1_vectorizer = str(config.MOD1_VECTORIZER_PATH)
    mod1_classifier = str(config.MOD1_CLASSIFIER_PATH)
    if not os.path.exists(mod1_vectorizer):
        errors.append(f"Module 1 vectorizer pickle not found at: {mod1_vectorizer}")
    if not os.path.exists(mod1_classifier):
        errors.append(f"Module 1 classifier pickle not found at: {mod1_classifier}")
        
    # Verify Module 2 Emotion Classification model adapter config
    mod2_dir = str(config.MOD2_DIR)
    mod2_config = str(config.MOD2_CONFIG_PATH)
    if not os.path.exists(mod2_dir):
        errors.append(f"Module 2 model directory not found at: {mod2_dir}")
    elif not os.path.exists(mod2_config):
        errors.append(f"Module 2 adapter config not found at: {mod2_config}")
        
    # Verify Qdrant database collection presence
    if qdrant_ok:
        try:
            from qdrant_client import QdrantClient
            if qdrant_url:
                client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                client = QdrantClient(path=qdrant_local_path)
            
            collections = [c.name for c in client.get_collections().collections]
            if config.QDRANT_COLLECTION_NAME not in collections:
                print(f"Notice: Qdrant collection '{config.QDRANT_COLLECTION_NAME}' not found. It will be created on startup.")
            else:
                count_info = client.count(collection_name=config.QDRANT_COLLECTION_NAME)
                if count_info.count == 0:
                    print(f"Notice: Qdrant collection '{config.QDRANT_COLLECTION_NAME}' is empty. It will be populated on startup.")
            client.close()
        except Exception as e:
            errors.append(f"Failed to connect to Qdrant or query collection: {e}")
            
    if errors:
        print("\n" + "="*80, file=sys.stderr)
        print("ENVIRONMENT VALIDATION FAILED on startup. Please resolve these issues:", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        print("="*80 + "\n", file=sys.stderr)
        sys.exit(1)


# ------------------------------------------------------------------------------
# 4. FastAPI Application Setup
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Serene AI - Empowering Mental Health Support API",
    description=(
        "🌿 **Serene AI API Engine**\n\n"
        "An advanced, multi-layered mental health support backend leveraging:\n"
        "- **BGE Reranker V2 M3** & BM25 Hybrid Retrieval\n"
        "- **XLM-RoBERTa** & custom local models for multilingual emotion/intent classification\n"
        "- **GPT OSS 20B** empathetic grounding\n\n"
        "All responses are clinically grounded in professional counseling case files."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

# Global RAG Instance
rag: MentalHealthRAG | None = None


# ------------------------------------------------------------------------------
# 5. Lifespan Event Listeners
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Runs concurrent startup model loading routines."""
    validate_environment()
    import asyncio
    
    async def load_rag():
        global rag
        rag = MentalHealthRAG()
        documents = await asyncio.to_thread(rag.load_and_preprocess)
        await asyncio.to_thread(rag.setup_retriever, documents)
        print("--> [RAG Setup] Vector store and retrievers preloaded successfully.")

    from .router import preload_models
    
    # Preload RAG and classifier/translator models concurrently
    await asyncio.gather(
        load_rag(),
        preload_models()
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Closes connections cleanly on shutdown."""
    global rag
    if rag is not None:
        rag.close()


# ------------------------------------------------------------------------------
# 6. HTTP Endpoints
# ------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serves the main Serene AI empathetic chat window UI."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health_check() -> dict:
    """System health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Processes queries through the routing and grounding RAG pipeline."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized.")

    result = await route_query(request.query, rag, history=request.history)
    if not result or "answer" not in result:
        raise HTTPException(status_code=500, detail="Failed to generate a response.")

    return ChatResponse(
        answer=result["answer"],
        resources=result.get("resources", []),
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
    )

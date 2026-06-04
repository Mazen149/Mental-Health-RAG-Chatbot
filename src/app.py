from dotenv import load_dotenv
load_dotenv()
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List

from .rag_pipeline import MentalHealthRAG
from .router import route_query


class ChatRequest(BaseModel):
    query: str


class Resource(BaseModel):
    score: float
    page_content: str
    response: str


class ChatResponse(BaseModel):
    answer: str
    resources: List[Resource]
    language: str | None = None
    emotion: List[str] | None = None
    intent: str | None = None


def validate_environment() -> None:
    import os
    import sys
    import importlib
    
    errors = []
    
    # 1. Environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    qdrant_url = os.getenv("QDRANT_URL")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    qdrant_local_path = os.path.join(base_dir, "qdrant_db")
    
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
        errors.append(f"Qdrant database not configured. Neither QDRANT_URL environment variable is set, nor does local Qdrant database folder exist at: {qdrant_local_path}")
        
    # 2. Required pip packages
    required_packages = [
        "peft", "transformers", "sentence_transformers", 
        "qdrant_client", "langchain_qdrant", "groq", "fastapi"
    ]
    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
        except ImportError:
            errors.append(f"Missing required pip package: {pkg}")
            
    # 3. Module 1 artifacts
    mod1_vectorizer = os.path.join(base_dir, "artifacts", "language_detection_best_vectorizer.pkl")
    mod1_classifier = os.path.join(base_dir, "artifacts", "language_detection_best_model.pkl")
    if not os.path.exists(mod1_vectorizer):
        errors.append(f"Module 1 vectorizer pickle not found at: {mod1_vectorizer}")
    if not os.path.exists(mod1_classifier):
        errors.append(f"Module 1 classifier pickle not found at: {mod1_classifier}")
        
    # 4. Module 2 model directory and adapter config
    mod2_dir = os.path.join(base_dir, "artifacts", "emotion_classifier")
    mod2_config = os.path.join(mod2_dir, "adapter_config.json")
    if not os.path.exists(mod2_dir):
        errors.append(f"Module 2 model directory not found at: {mod2_dir}")
    elif not os.path.exists(mod2_config):
        errors.append(f"Module 2 adapter config not found at: {mod2_config}")
        
    # 5. Qdrant collection check
    if qdrant_ok:
        try:
            from qdrant_client import QdrantClient
            if qdrant_url:
                client = QdrantClient(url=qdrant_url)
            else:
                client = QdrantClient(path=qdrant_local_path)
            
            collections = [c.name for c in client.get_collections().collections]
            if "mental_health" not in collections:
                errors.append("Qdrant collection 'mental_health' not found in database.")
            else:
                count_info = client.count(collection_name="mental_health")
                if count_info.count == 0:
                    errors.append("Qdrant collection 'mental_health' is empty (contains 0 vectors).")
            client.close()
        except Exception as e:
            errors.append(f"Failed to connect to Qdrant or query collection 'mental_health': {e}")
            
    if errors:
        print("\n" + "="*80, file=sys.stderr)
        print("ENVIRONMENT VALIDATION FAILED on startup. Please resolve the following issues:", file=sys.stderr)
        for err in errors:
            print(f" - {err}", file=sys.stderr)
        print("="*80 + "\n", file=sys.stderr)
        sys.exit(1)


app = FastAPI(
    title="Mental Health RAG Chatbot",
    description="A FastAPI chatbot that uses MentalHealthRAG for retrieval-augmented mental health responses.",
    version="0.1.0",
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

rag: MentalHealthRAG | None = None


@app.on_event("startup")
async def startup_event() -> None:
    validate_environment()
    global rag
    rag = MentalHealthRAG()
    documents = rag.load_and_preprocess()
    rag.setup_retriever(documents)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global rag
    if rag is not None:
        rag.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized.")

    result = await route_query(request.query, rag)
    if not result or "answer" not in result:
        raise HTTPException(status_code=500, detail="Failed to generate a response.")

    return ChatResponse(
        answer=result["answer"],
        resources=result.get("resources", []),
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
    )

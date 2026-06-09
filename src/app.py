"""
================================================================================
SERENE AI — FASTAPI APPLICATION SERVER
================================================================================
Empathetic, multi-layered mental health support backend leveraging
asynchronous model loading, hybrid retrieval (BM25 + Qdrant), and LLM grounding.
================================================================================
"""

import importlib
import asyncio
import json
import hashlib
import os
import hmac
import sqlite3
from pathlib import Path
import sys
from typing import Any, List


from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

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


class HistoryItem(BaseModel):
    role: str
    content: str
    language: str | None = None
    emotion: List[str] | None = None
    intent: str | None = None
    resources: list | None = None


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

app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET_KEY,
    same_site="lax",
    https_only=False,
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
DB_PATH = Path(config.CHAT_DATABASE_PATH)


def _is_authenticated(request: Request) -> bool:
    return bool(request.session.get("authenticated"))


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), password_hash.hex()


def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    _, computed_hash_hex = _hash_password(password, salt)
    return hmac.compare_digest(computed_hash_hex, expected_hash_hex)


def _init_chat_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                language TEXT,
                emotion_json TEXT,
                intent TEXT,
                resources_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_interactions_user_id ON chat_interactions(user_id)"
        )
        conn.commit()


def _db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _get_user_by_username(username: str) -> dict[str, Any] | None:
    with _db_connection() as conn:
        row = conn.execute(
            """
            SELECT id, username, password_salt, password_hash, created_at, last_login_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    return dict(row) if row else None


def _create_user(username: str, password: str) -> dict[str, Any]:
    salt_hex, password_hash_hex = _hash_password(password)
    with _db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (username, password_salt, password_hash)
            VALUES (?, ?, ?)
            """,
            (username, salt_hex, password_hash_hex),
        )
        conn.commit()
        row = conn.execute(
            """
            SELECT id, username, password_salt, password_hash, created_at, last_login_at
            FROM users
            WHERE id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row) if row else {
            "id": cursor.lastrowid,
            "username": username,
            "password_salt": salt_hex,
            "password_hash": password_hash_hex,
        }


def _update_last_login(user_id: int) -> None:
    with _db_connection() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,),
        )
        conn.commit()


def _save_chat_interaction(
    *,
    user_id: int,
    query: str,
    response: str,
    language: str | None = None,
    emotion: list[str] | None = None,
    intent: str | None = None,
    resources: list | None = None,
) -> None:
    emotion_json = json.dumps(emotion, ensure_ascii=False) if emotion is not None else None
    resources_json = json.dumps(resources, ensure_ascii=False) if resources is not None else None
    with _db_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_interactions (
                user_id, query, response, language, emotion_json, intent, resources_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, query, response, language, emotion_json, intent, resources_json),
        )
        conn.commit()


def _load_chat_history(user_id: int) -> list[dict[str, Any]]:
    with _db_connection() as conn:
        rows = conn.execute(
            """
            SELECT query, response, language, emotion_json, intent, resources_json
            FROM chat_interactions
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,),
        ).fetchall()

    history: list[dict[str, Any]] = []
    for row in rows:
        emotion = json.loads(row["emotion_json"]) if row["emotion_json"] else None
        resources = json.loads(row["resources_json"]) if row["resources_json"] else None
        history.append(
            {
                "role": "user",
                "content": row["query"],
                "language": row["language"],
                "emotion": None,
                "intent": None,
                "resources": None,
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": row["response"],
                "language": row["language"],
                "emotion": emotion,
                "intent": row["intent"],
                "resources": resources,
            }
        )
    return history


def _split_stream_chunks(text: str, chunk_size: int = 24) -> list[str]:
    words = text.split()
    if not words:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def _sse_event(event: str, data: Any) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    payload = payload.replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {payload}\n\n"


# ------------------------------------------------------------------------------
# 5. Lifespan Event Listeners
# ------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    """Runs concurrent startup model loading routines."""
    _init_chat_db()
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
    if not _is_authenticated(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "mode": "chat",
            "username": request.session.get("username", "Guest"),
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "mode": "login",
            "error": None,
            "success": request.query_params.get("registered") == "1",
        },
    )


@app.post("/login")
async def login(request: Request) -> HTMLResponse:
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", "")).strip()

    if not username or not password:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "login",
                "error": "Please enter both a username and password.",
                "success": False,
            },
            status_code=401,
        )

    user = _get_user_by_username(username)
    if user is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "login",
                "error": "No account found for that username. Please register first.",
                "success": False,
            },
            status_code=401,
        )
    elif not _verify_password(password, user["password_salt"], user["password_hash"]):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "login",
                "error": "Invalid username or password.",
                "success": False,
            },
            status_code=401,
        )

    request.session["authenticated"] = True
    request.session["user_id"] = int(user["id"])
    request.session["username"] = username
    _update_last_login(int(user["id"]))
    return RedirectResponse(url="/", status_code=302)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    if _is_authenticated(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "mode": "register",
            "error": None,
        },
    )


@app.post("/register")
async def register(request: Request) -> HTMLResponse:
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", "")).strip()
    confirm_password = str(form.get("confirm_password", "")).strip()

    if not username or not password or not confirm_password:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "register",
                "error": "Please fill in every field.",
            },
            status_code=400,
        )

    if password != confirm_password:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "register",
                "error": "Passwords do not match. Please try again.",
            },
            status_code=400,
        )

    if _get_user_by_username(username) is not None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "register",
                "error": "That username already exists. Please choose another one or log in.",
            },
            status_code=400,
        )

    try:
        _create_user(username, password)
    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "mode": "register",
                "error": "Could not create your account. Please try again.",
            },
            status_code=500,
        )

    return RedirectResponse(url="/login?registered=1", status_code=302)


@app.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


@app.get("/health")
async def health_check() -> dict:
    """System health check endpoint."""
    return {"status": "ok"}


@app.get("/chat/history")
async def chat_history(request: Request) -> list[dict[str, Any]]:
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")

    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    return _load_chat_history(int(user_id))


@app.post("/chat/clear")
async def clear_chat(request: Request) -> dict:
    if not _is_authenticated(request):
        raise HTTPException(status_code=401, detail="Authentication required.")
    user_id = request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    with _db_connection() as conn:
        conn.execute("DELETE FROM chat_interactions WHERE user_id = ?", (int(user_id),))
        conn.commit()
    return {"status": "ok", "message": "Chat history cleared successfully."}



@app.post("/chat", response_model=ChatResponse)
async def chat(page_request: Request, request: ChatRequest) -> ChatResponse:
    """Processes queries through the routing and grounding RAG pipeline."""
    if not _is_authenticated(page_request):
        raise HTTPException(status_code=401, detail="Authentication required.")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized.")

    user_id = page_request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    result: dict[str, Any] | None = None
    error_response: str | None = None
    try:
        result = await route_query(request.query, rag, history=request.history)
    except Exception:
        error_response = "Failed to generate a response."

    if not result or "answer" not in result:
        _save_chat_interaction(
            user_id=int(user_id),
            query=request.query.strip(),
            response=error_response or "Failed to generate a response.",
        )
        raise HTTPException(status_code=500, detail=error_response or "Failed to generate a response.")

    _save_chat_interaction(
        user_id=int(user_id),
        query=request.query.strip(),
        response=str(result["answer"]),
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
        resources=result.get("resources", []),
    )

    return ChatResponse(
        answer=result["answer"],
        resources=result.get("resources", []),
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
    )


@app.post("/chat/stream")
async def chat_stream(page_request: Request, request: ChatRequest) -> StreamingResponse:
    """Streams the generated answer as SSE chunks after the RAG response is ready."""
    if not _is_authenticated(page_request):
        raise HTTPException(status_code=401, detail="Authentication required.")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized.")

    user_id = page_request.session.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required.")

    try:
        result = await route_query(request.query, rag, history=request.history)
    except Exception:
        result = {"answer": "Failed to generate a response.", "resources": []}

    answer = str(result.get("answer", "")).strip() or "Failed to generate a response."
    resources = result.get("resources", [])
    _save_chat_interaction(
        user_id=int(user_id),
        query=request.query.strip(),
        response=answer,
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
        resources=resources,
    )

    async def event_generator():
        yield _sse_event("meta", {
            "language": result.get("language"),
            "emotion": result.get("emotion"),
            "intent": result.get("intent"),
            "resources": resources,
        })
        yield _sse_event("start", {"status": "streaming"})

        for chunk in _split_stream_chunks(answer, chunk_size=18):
            yield _sse_event("chunk", {"text": chunk})
            await asyncio.sleep(0.02)

        yield _sse_event("done", {"status": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/transcribe")
async def transcribe(page_request: Request, file: UploadFile = File(...)) -> dict:
    """
    Transcribes speech input using Groq's Whisper model (whisper-large-v3).
    """
    if not _is_authenticated(page_request):
        raise HTTPException(status_code=401, detail="Authentication required.")

    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="Groq API key is not configured.")

    try:
        from groq import Groq
        # Initialize Groq client
        client = Groq(api_key=config.GROQ_API_KEY)
        
        file_bytes = await file.read()
        filename = file.filename or "recording.wav"
        
        # Call Groq's speech-to-text API
        transcription = client.audio.transcriptions.create(
            file=(filename, file_bytes),
            model="whisper-large-v3",
        )
        return {"text": transcription.text}
    except Exception as e:
        print(f"--> [Speech-to-Text Error] Audio transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

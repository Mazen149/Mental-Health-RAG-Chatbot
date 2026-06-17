"""
================================================================================
SANAD AI — FASTAPI APPLICATION SERVER
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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# OpenTelemetry & HyperDX Configuration
# ------------------------------------------------------------------------------
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

hyperdx_api_key = os.getenv("HYPERDX_API_KEY", "")

_resource = Resource.create({
    "service.name": "serenity-api",
    "service.version": "1.0.0",
})

if hyperdx_api_key:
    # Use HyperDX OTLP endpoint
    _metric_exporter = OTLPMetricExporter(
        endpoint="https://in.hyperdx.io/v1/metrics",
        headers={"Authorization": hyperdx_api_key}
    )
    _span_exporter = OTLPSpanExporter(
        endpoint="https://in.hyperdx.io/v1/traces",
        headers={"Authorization": hyperdx_api_key}
    )
else:
    # Fallback to local collector
    _otel_endpoint_metrics = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/metrics")
    _otel_endpoint_traces = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
    _metric_exporter = OTLPMetricExporter(endpoint=_otel_endpoint_metrics)
    _span_exporter = OTLPSpanExporter(endpoint=_otel_endpoint_traces)

# Set up Metrics
_metric_reader = PeriodicExportingMetricReader(_metric_exporter, export_interval_millis=15000)
_meter_provider = MeterProvider(resource=_resource, metric_readers=[_metric_reader])
metrics.set_meter_provider(_meter_provider)

# Set up Traces (Spans)
_tracer_provider = TracerProvider(resource=_resource)
_tracer_provider.add_span_processor(BatchSpanProcessor(_span_exporter))
trace.set_tracer_provider(_tracer_provider)

# Create a Meter
meter = metrics.get_meter("serenity_meter", version="1.0.0")

intent_counter = meter.create_counter(
    name="serenity.intent.count",
    description="Count of classified intents",
    unit="1",
)

message_length_histogram = meter.create_histogram(
    name="serenity.message.length",
    description="Character length of user messages",
    unit="characters",
)

http_request_counter = meter.create_counter(
    name="serenity.http.requests",
    description="Count of HTTP requests by endpoint and status",
    unit="1",
)

feedback_counter = meter.create_counter(
    name="serenity.feedback.votes",
    description="Count of user feedback votes",
    unit="1",
)

# Start System Metrics (CPU/RAM) Collection
SystemMetricsInstrumentor().instrument()

# ------------------------------------------------------------------------------
# 1. Environment Loading & Configuration
# ------------------------------------------------------------------------------
from .config import config
from . import metrics

# Local project imports (must load after environment resolution)
from .modules.rag import MentalHealthRAG
from .router import route_query


# ------------------------------------------------------------------------------
# 2. Pydantic API Schemas
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(
        ..., description="The role of the message sender: 'user' or 'assistant'."
    )
    content: str = Field(..., description="The text content of the message.")


class ChatRequest(BaseModel):
    query: str | None = Field(
        None, description="The user's query or message to the mental health chatbot."
    )
    message: str | None = Field(
        None, description="Alternative field for user message/query."
    )
    history: List[ChatMessage] | None = Field(
        None, description="The recent chat history turns for conversational context."
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
        description="The semantic or reranked relevance score of this document context.",
    )
    page_content: str = Field(
        ..., description="The counseling context representing the user's situation."
    )
    response: str = Field(
        ...,
        description="The verified clinical advice or response corresponding to this context.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(
        ...,
        description="The empathetic, model-generated response grounded in grounding contexts.",
    )
    resources: List[Resource] = Field(
        ...,
        description="The list of grounding counseling cases and clinician advices retrieved.",
    )
    language: str | None = Field(
        None, description="The detected language of the user query."
    )
    emotion: List[str] | None = Field(
        None, description="The detected emotional states from emotion classifier."
    )
    intent: str | None = Field(
        None,
        description="The classified conversational or clinical intent of the query.",
    )


class FeedbackRequest(BaseModel):
    vote: str = Field(
        ..., description="The user feedback vote, exactly 'up' or 'down'."
    )
    user_message: str = Field(
        ..., description="The user query associated with this feedback."
    )
    bot_response: str = Field(
        ..., description="The bot response associated with this feedback."
    )


class UserAuthRequest(BaseModel):
    username: str = Field(..., description="The user's username.")
    password: str = Field(..., description="The user's password.")


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
        "fastembed",
        "onnxruntime",
        "qdrant_client",
        "langchain_qdrant",
        "groq",
        "fastapi",
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
                logger.info(
                    f"Notice: Qdrant collection '{config.QDRANT_COLLECTION_NAME}' not found. It will be created on startup."
                )
            else:
                count_info = client.count(collection_name=config.QDRANT_COLLECTION_NAME)
                if count_info.count == 0:
                    logger.info(
                        f"Notice: Qdrant collection '{config.QDRANT_COLLECTION_NAME}' is empty. It will be populated on startup."
                    )
            client.close()
        except Exception as e:
            errors.append(f"Failed to connect to Qdrant or query collection: {e}")

    if errors:
        logger.error("\n" + "=" * 80)
        logger.error(
            "ENVIRONMENT VALIDATION FAILED on startup. Please resolve these issues:"
        )
        for err in errors:
            logger.error(f" - {err}")
        logger.error("=" * 80 + "\n")
        sys.exit(1)


# ------------------------------------------------------------------------------
# 4. FastAPI Application Setup
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Sanad AI - Empowering Mental Health Support API",
    description=(
        "🌿 **Sanad AI API Engine**\n\n"
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

is_production = bool(config.TURSO_DATABASE_URL)

FastAPIInstrumentor.instrument_app(app)

@app.middleware("http")
async def track_requests(request: Request, call_next):
    response = await call_next(request)
    http_request_counter.add(1, {
        "method": request.method,
        "endpoint": request.url.path,
        "status_code": str(response.status_code),
    })
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|.*\.github\.io)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET_KEY,
    same_site=config.SESSION_COOKIE_SAMESITE,
    https_only=config.SESSION_COOKIE_SECURE,
)

# Global RAG Instance
rag: MentalHealthRAG | None = None
DB_PATH = Path(config.CHAT_DATABASE_PATH)


# ------------------------------------------------------------------------------
# LibSQL Compatibility Wrappers for Turso (matching sqlite3.Row / sqlite3 API)
# ------------------------------------------------------------------------------
class LibsqlRow:
    def __init__(self, description, row_tuple):
        self._keys = [col[0] for col in description] if description else []
        self._values = row_tuple
        self._dict = dict(zip(self._keys, self._values))

    def keys(self) -> list[str]:
        return self._keys

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._dict[key]


class LibsqlCursorWrapper:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, *args, **kwargs):
        self._cursor.execute(*args, **kwargs)
        return self

    def executemany(self, *args, **kwargs):
        self._cursor.executemany(*args, **kwargs)
        return self

    def executescript(self, *args, **kwargs):
        self._cursor.executescript(*args, **kwargs)
        return self

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def description(self):
        return self._cursor.description

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return LibsqlRow(self._cursor.description, row)

    def fetchall(self):
        rows = self._cursor.fetchall()
        desc = self._cursor.description
        return [LibsqlRow(desc, r) for r in rows]

    def fetchmany(self, size=None):
        if size is None:
            rows = self._cursor.fetchmany()
        else:
            rows = self._cursor.fetchmany(size)
        desc = self._cursor.description
        return [LibsqlRow(desc, r) for r in rows]

    def close(self):
        self._cursor.close()

    def __iter__(self):
        desc = self._cursor.description
        for r in self._cursor:
            yield LibsqlRow(desc, r)


class LibsqlConnectionWrapper:
    def __init__(self, connection):
        self._conn = connection

    def cursor(self):
        return LibsqlCursorWrapper(self._conn.cursor())

    def execute(self, *args, **kwargs):
        cur = self.cursor()
        cur.execute(*args, **kwargs)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 200_000
    )
    return salt.hex(), password_hash.hex()


def _verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    _, computed_hash_hex = _hash_password(password, salt)
    return hmac.compare_digest(computed_hash_hex, expected_hash_hex)


def _is_authenticated(request: Request) -> bool:
    """Checks if the user is authenticated in the current session."""
    return request.session.get("authenticated", False)


def _get_active_user_id(request: Request) -> int:
    """Fetches the user_id from the session if authenticated; otherwise falls back to guest user_id."""
    if _is_authenticated(request):
        user_id = request.session.get("user_id")
        if user_id is not None:
            return int(user_id)
    return _get_or_create_guest_user_id()


def _db_connection() -> Any:
    if config.TURSO_DATABASE_URL:
        import libsql
        conn = libsql.connect(
            database=config.TURSO_DATABASE_URL,
            auth_token=config.TURSO_AUTH_TOKEN or "",
        )
        return LibsqlConnectionWrapper(conn)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _init_chat_db() -> None:
    if not config.TURSO_DATABASE_URL:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_connection() as conn:
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                vote TEXT NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _save_feedback(
    user_id: int | None, vote: str, user_message: str, bot_response: str
) -> None:
    with _db_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (user_id, vote, user_message, bot_response)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, vote, user_message, bot_response),
        )
        conn.commit()


def _get_or_create_guest_user_id() -> int:
    with _db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", ("guest",))
        row = cursor.fetchone()
        if row:
            return int(row["id"])
        # Create guest user
        salt, pwhash = _hash_password("guest_pass_123_xyz")
        cursor.execute(
            """
            INSERT INTO users (username, password_salt, password_hash)
            VALUES (?, ?, ?)
            """,
            ("guest", salt, pwhash),
        )
        conn.commit()
        return cursor.lastrowid


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
        return (
            dict(row)
            if row
            else {
                "id": cursor.lastrowid,
                "username": username,
                "password_salt": salt_hex,
                "password_hash": password_hash_hex,
            }
        )


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
    emotion_json = (
        json.dumps(emotion, ensure_ascii=False) if emotion is not None else None
    )
    resources_json = (
        json.dumps(resources, ensure_ascii=False) if resources is not None else None
    )
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
    # Initialize metrics (OpenTelemetry)
    try:
        metrics.init_metrics()
        print("--> [Metrics] OpenTelemetry metrics initialized.")
    except Exception as e:
        print(f"--> [Metrics] Failed to initialize metrics: {e}")

    _init_chat_db()

    # Download missing artifacts from Hugging Face if needed
    from .modules.downloader import download_artifacts

    download_artifacts()

    validate_environment()
    import asyncio

    async def load_rag():
        global rag
        _temp_rag = MentalHealthRAG()
        documents = await asyncio.to_thread(_temp_rag.load_and_preprocess)
        await asyncio.to_thread(_temp_rag.setup_retriever, documents)
        rag = _temp_rag
        logger.info(
            "--> [RAG Setup] Vector store and retrievers preloaded successfully."
        )

    from .router import preload_models

    # Preload RAG and classifier/translator models concurrently as background tasks.
    # This prevents the Azure health check from timing out while indexing vectors.
    asyncio.create_task(load_rag())
    asyncio.create_task(preload_models())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Closes connections cleanly on shutdown."""
    global rag
    if rag is not None:
        rag.close()


# ------------------------------------------------------------------------------
# 6. HTTP Endpoints
# ------------------------------------------------------------------------------


@app.get("/")
@app.get("/health")
async def health_check() -> dict:
    """System health check endpoint."""
    return {"status": "ok"}


@app.get("/chat/history")
async def chat_history(request: Request) -> list[dict[str, Any]]:
    if not _is_authenticated(request):
        raise HTTPException(
            status_code=401, detail="Authentication required. Please login first."
        )
    user_id = request.session.get("user_id")
    return _load_chat_history(int(user_id))


@app.post("/chat/clear")
async def clear_chat(request: Request) -> dict:
    if not _is_authenticated(request):
        raise HTTPException(
            status_code=401, detail="Authentication required. Please login first."
        )
    user_id = request.session.get("user_id")
    with _db_connection() as conn:
        conn.execute("DELETE FROM chat_interactions WHERE user_id = ?", (int(user_id),))
        conn.commit()
    return {"status": "ok", "message": "Chat history cleared successfully."}


@app.post("/chat", response_model=ChatResponse)
async def chat(page_request: Request, request: ChatRequest) -> ChatResponse:
    """Processes queries through the routing and grounding RAG pipeline."""
    if not _is_authenticated(page_request):
        raise HTTPException(
            status_code=401, detail="Authentication required. Please login first."
        )
    user_id = page_request.session.get("user_id")

    query_text = (request.query or request.message or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is initializing in the background. Please wait a few moments.")

    # Metrics: increment request and record message length
    try:
        metrics.record_request()
        metrics.record_message_length(len(query_text))
    except Exception:
        pass

    result: dict[str, Any] | None = None
    error_response: str | None = None
    try:
        import time

        start = time.time()
        result = await route_query(query_text, rag, history=request.history)
        elapsed_ms = (time.time() - start) * 1000.0
        try:
            metrics.record_response_latency(elapsed_ms)
        except Exception:
            pass
    except Exception:
        try:
            metrics.record_error()
        except Exception:
            pass
        error_response = "Failed to generate a response."

    if not result or "answer" not in result:
        _save_chat_interaction(
            user_id=int(user_id),
            query=query_text,
            response=error_response or "Failed to generate a response.",
        )
        raise HTTPException(
            status_code=500, detail=error_response or "Failed to generate a response."
        )
    
    _save_chat_interaction(
        user_id=int(user_id),
        query=query_text,
        response=str(result["answer"]),
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
        resources=result.get("resources", []),
    )

    # Record OTel metrics
    message_length_histogram.record(len(query_text))
    if result and "intent" in result:
        intent_counter.add(1, {"intent": result.get("intent", "unknown")})

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
        raise HTTPException(
            status_code=401, detail="Authentication required. Please login first."
        )
    user_id = page_request.session.get("user_id")

    query_text = (request.query or request.message or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required.")

    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine is not initialized.")

    try:
        result = await route_query(query_text, rag, history=request.history)
    except Exception:
        result = {"answer": "Failed to generate a response.", "resources": []}

    answer = str(result.get("answer", "")).strip() or "Failed to generate a response."
    resources = result.get("resources", [])
    
    _save_chat_interaction(
        user_id=int(user_id),
        query=query_text,
        response=answer,
        language=result.get("language"),
        emotion=result.get("emotion"),
        intent=result.get("intent"),
        resources=resources,
    )

    async def event_generator():
        yield _sse_event(
            "meta",
            {
                "language": result.get("language"),
                "emotion": result.get("emotion"),
                "intent": result.get("intent"),
            },
        )
        yield _sse_event("start", {"status": "streaming"})

        for chunk in _split_stream_chunks(answer, chunk_size=1):
            yield _sse_event("chunk", {"text": chunk})
            await asyncio.sleep(0.03)

        yield _sse_event("citations", {"resources": resources})
        yield _sse_event("done", {"status": "done"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/transcribe")
async def transcribe(page_request: Request, file: UploadFile = File(...)) -> dict:
    """
    Transcribes speech input using Groq's Whisper model (whisper-large-v3).
    """
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
        logger.error(f"--> [Speech-to-Text Error] Audio transcription failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to transcribe audio: {str(e)}"
        )


@app.post("/feedback")
async def save_feedback(page_request: Request, request: FeedbackRequest) -> dict:
    """Saves user thumbs up/down feedback on bot replies."""
    if request.vote not in ("up", "down"):
        raise HTTPException(status_code=400, detail="Vote must be 'up' or 'down'.")

    user_id = _get_active_user_id(page_request)
    _save_feedback(
        user_id=int(user_id) if user_id is not None else None,
        vote=request.vote,
        user_message=request.user_message.strip(),
        bot_response=request.bot_response.strip(),
    )
    try:
        metrics.record_feedback_vote(request.vote)
    except Exception:
        pass
    return {"status": "ok", "message": "Feedback saved successfully."}


@app.post("/register")
async def register(request: Request, auth_payload: UserAuthRequest) -> dict:
    username = auth_payload.username.strip()
    password = auth_payload.password.strip()

    if not username or not password:
        raise HTTPException(
            status_code=400, detail="Username and password cannot be empty."
        )

    # Check if user already exists
    existing_user = _get_user_by_username(username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered.")

    # Create new user
    user = _create_user(username, password)

    # Save to session
    request.session["authenticated"] = True
    request.session["user_id"] = user["id"]

    return {
        "status": "ok",
        "message": "User registered successfully.",
        "username": username,
    }


@app.post("/login")
async def login(request: Request, auth_payload: UserAuthRequest) -> dict:
    username = auth_payload.username.strip()
    password = auth_payload.password.strip()

    if not username or not password:
        raise HTTPException(
            status_code=400, detail="Username and password cannot be empty."
        )

    user = _get_user_by_username(username)
    if not user or not _verify_password(
        password, user["password_salt"], user["password_hash"]
    ):
        raise HTTPException(
            status_code=401, detail="Incorrect username or password."
        )

    # Save to session
    request.session["authenticated"] = True
    request.session["user_id"] = user["id"]
    _update_last_login(user["id"])

    return {
        "status": "ok",
        "message": "Logged in successfully.",
        "username": username,
    }


@app.post("/logout")
async def logout(request: Request) -> dict:
    request.session.clear()
    return {"status": "ok", "message": "Logged out successfully."}

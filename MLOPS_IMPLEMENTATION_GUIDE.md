# Serenity MLOps — Complete Step-by-Step Implementation Guide

> **Purpose**: This document is a self-contained, educational, step-by-step guide for completing
> the MLOps Final Project for the Serenity mental health support chatbot. It is designed to be
> read by both humans and LLMs so that any AI assistant can pick up where you left off.
>
> **Project Root**: `E:/ITI/NLP/project/Mental-Health-RAG-Chatbot`
> **Frontend Root**: `E:/ITI/NLP/project/chatbot-frontend`
> **Branch**: `ahmed-backend`
> **Date Created**: 2026-06-16

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Current State — What Is Already Done](#2-current-state--what-is-already-done)
3. [Rubric Checklist](#3-rubric-checklist)
4. [Phase 1: Logging (DONE)](#4-phase-1-logging-done)
5. [Phase 2: Rate Limiting (DONE)](#5-phase-2-rate-limiting-done)
6. [Phase 3: OpenTelemetry Metrics](#6-phase-3-opentelemetry-metrics)
7. [Phase 4: OpenTelemetry Collector & Axiom](#7-phase-4-opentelemetry-collector--axiom)
8. [Phase 5: CI/CD Pipeline](#8-phase-5-cicd-pipeline)
9. [Phase 6: Azure Deployment](#9-phase-6-azure-deployment)
10. [Phase 7: Frontend Integration](#10-phase-7-frontend-integration)
11. [Phase 8: README Updates](#11-phase-8-readme-updates)
12. [Phase 9: Model Comparison (Bonus)](#12-phase-9-model-comparison-bonus)
13. [Phase 10: Demo Video](#13-phase-10-demo-video)
14. [Final Deliverables Checklist](#14-final-deliverables-checklist)
15. [Appendix A: Azure Free Tier Analysis](#appendix-a-azure-free-tier-analysis)
16. [Appendix B: Troubleshooting](#appendix-b-troubleshooting)

---

## 1. Project Overview & Architecture

### 1.1 What is Serenity?

Serenity (also called Sanad) is a **mental health support chatbot** that:
- Classifies user **intent** (greeting, goodbye, mental health question, crisis, out-of-scope)
- Detects user **emotion** (sadness, anxiety, anger, etc.) using a fine-tuned XLM-RoBERTa model
- Detects user **language** (20+ languages) using a TF-IDF classifier
- Retrieves relevant **counseling contexts** from a Qdrant vector database (RAG)
- Generates **empathetic responses** grounded in those contexts using Groq's GPT-OSS-20B LLM

### 1.2 Repository Structure

```
Mental-Health-RAG-Chatbot/
├── .dockerignore
├── .env                         # Your API keys (never commit this)
├── .env.example                 # Template for .env
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # [TO CREATE] GitHub Actions pipeline
├── .gitignore
├── .pre-commit-config.yaml      # Ruff linter + formatter hooks
├── Dockerfile                   # Docker containerization
├── README.md                    # Project documentation
├── main.py                      # Dev server entry point
├── otel-collector-config.yaml   # [TO CREATE] OpenTelemetry Collector config
├── pyproject.toml               # Dependencies & project metadata
├── uv.lock                      # Locked dependency versions
├── src/
│   ├── __init__.py
│   ├── app.py                   # FastAPI application (main file)
│   ├── config.py                # Centralized configuration
│   ├── router.py                # Query routing engine
│   └── modules/
│       ├── __init__.py          # Exposes detect_language, classify_emotion, classify_intent
│       ├── downloader.py        # HuggingFace artifact downloader
│       ├── emotion_classifier.py
│       ├── intent_classifier.py
│       ├── language_detector.py
│       ├── multilingual_patterns.py  # Regex patterns for 20 languages
│       ├── rag.py               # RAG engine (retrieval + generation)
│       └── rag_evaluation.py    # RAGAS/DeepEval evaluation
├── tests/
│   ├── test_emotion_classifier.py
│   ├── test_feedback.py
│   ├── test_intent_classifier.py
│   ├── test_language_detector.py
│   ├── test_mental_health_rag.py
│   └── test_router.py
└── artifacts/                   # Model weights, pickles, databases
```

### 1.3 API Endpoints

| Method | Path | Purpose | Auth Required |
| :--- | :--- | :--- | :---: |
| `GET` | `/health` | Health check | No |
| `POST` | `/chat` | Send message, get response | No (guest fallback) |
| `POST` | `/chat/stream` | SSE streaming version of /chat | No (guest fallback) |
| `POST` | `/feedback` | Submit thumbs up/down | No (guest fallback) |
| `GET` | `/chat/history` | Load chat history | Yes (session) |
| `POST` | `/chat/clear` | Clear chat history | Yes (session) |
| `POST` | `/login` | Login (form) | No |
| `POST` | `/register` | Register (form) | No |
| `POST` | `/logout` | Logout | Yes (session) |
| `POST` | `/transcribe` | Speech-to-text | Yes (session) |
| `GET` | `/docs` | Swagger UI | No |

### 1.4 Frontend API Contract

The frontend (`chatbot-frontend/app.js`) calls exactly two endpoints:

**1. POST /chat**
```json
// Request:
{ "message": "I've been feeling anxious" }

// Response:
{
  "answer": "I hear you. Anxiety can feel...",
  "resources": [...],
  "language": "English",
  "emotion": ["Anxiety"],
  "intent": "asking_mental_health_question"
}
```

**2. POST /feedback**
```json
// Request:
{ "vote": "up", "user_message": "...", "bot_response": "..." }

// Response:
{ "status": "ok", "message": "Feedback saved successfully." }
```

---

## 2. Current State — What Is Already Done

### Completed

- **Repository setup**: Clean structure, `.pre-commit-config.yaml` with ruff
- **NLP pipeline**: Intent classification, emotion detection, language detection, RAG, crisis handling
- **Model serving**: FastAPI with `/chat`, `/feedback`, `/health`, CORS, guest auth fallback
- **Unit tests**: 6 test files, 27 core tests passing
- **Containerization**: Dockerfile with `python:3.12-slim`, `uv sync --frozen`, layer caching
- **Logging**: All `print()` replaced with Python `logging` module (`INFO`, `WARNING`, `ERROR`)
- **Rate limiting**: `slowapi` wired into `/chat` (10/min), `/chat/stream` (10/min), `/feedback` (20/min)

### Remaining

- OpenTelemetry metrics instrumentation
- OTel Collector + Axiom setup
- CI/CD GitHub Actions workflow
- Azure Container Apps deployment
- Frontend API URL update to deployed URL
- README sections (metrics explanation, dashboard screenshot, deployed URL)
- Model comparison table/graph (bonus)
- Demo video

---

## 3. Rubric Checklist

| # | Section | Points | Status |
| :---: | :--- | :---: | :---: |
| 1 | Repository Setup | 5 | DONE |
| 2 | NLP Pipeline | — | DONE |
| 3 | Model Serving | 10 | DONE |
| 4 | Unit Testing | 10 | DONE |
| 5 | Containerization | 10 | DONE |
| 6 | Monitoring Metrics | 10 | TODO |
| 7 | System Monitoring | 10 | TODO |
| 8 | CI/CD Pipeline | 10 | TODO |
| 9 | Deployment | 10 | TODO |
| 10 | Frontend Integration | 10 | PARTIAL |
| 11 | Bonus: Rate Limiting | 5 | DONE |
| 12 | Bonus: Model Comparison | 5 | TODO |
| | **Total possible** | **95+10** | |

---

## 4. Phase 1: Logging (DONE)

### 4.1 What Is Logging?

**Logging** is the practice of recording events, errors, and informational messages while an
application runs. Python's built-in `logging` module provides a standardized way to do this
with severity levels.

### 4.2 Why Not Just Use print()?

| Feature | `print()` | `logging` |
| :--- | :--- | :--- |
| Severity levels | No | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| Timestamps | No | Automatic |
| Module/source info | No | Shows which file/logger |
| Configurable output | No | Can redirect to files, services |
| Production-ready | No | Industry standard |
| Filterable | No | Can suppress INFO in production |

### 4.3 What We Did

**In `src/app.py`:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("serenity_api")
```

Then replaced all `print(...)` with `logger.info(...)` or `logger.error(...)`.

**In `src/router.py`:**
```python
import logging
logger = logging.getLogger("serenity.router")

def safe_print(msg: str, level: str = "info") -> None:
    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)
```

### 4.4 Log Level Cheat Sheet

| Level | When to use | Example |
| :--- | :--- | :--- |
| `DEBUG` | Detailed diagnostic info (hidden in production) | `logger.debug(f"Query tokens: {tokens}")` |
| `INFO` | Confirming things work as expected | `logger.info("RAG engine loaded")` |
| `WARNING` | Something unexpected but not broken | `logger.warning("Qdrant collection empty")` |
| `ERROR` | A serious problem occurred | `logger.error(f"LLM call failed: {e}")` |
| `CRITICAL` | The app might crash | `logger.critical("Database corrupted")` |

---

## 5. Phase 2: Rate Limiting (DONE)

### 5.1 What Is Rate Limiting?

Rate limiting restricts how many requests a single client (identified by IP address) can make
to an endpoint within a time window. This prevents:
- **Abuse**: Someone spamming your chatbot with thousands of requests
- **Cost overruns**: Each `/chat` call hits the Groq LLM API (which has its own rate limits)
- **DoS attacks**: Overwhelming your server with traffic

### 5.2 How slowapi Works

```
User sends request -> slowapi checks IP address -> counts recent requests ->
  IF count < limit -> allow request through
  IF count >= limit -> return HTTP 429 "Too Many Requests"
```

### 5.3 What We Added

**Dependencies** (in `pyproject.toml`):
```toml
"slowapi>=0.1.9",
```

**In `src/app.py`:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# After app = FastAPI(...)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# On endpoints:
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(page_request: Request, request: ChatRequest) -> ChatResponse:
    ...

@app.post("/chat/stream")
@limiter.limit("10/minute")
async def chat_stream(page_request: Request, request: ChatRequest) -> StreamingResponse:
    ...

@app.post("/feedback")
@limiter.limit("20/minute")
async def save_feedback(page_request: Request, request: FeedbackRequest) -> dict:
    ...
```

### 5.4 How the Limits Work

- `"10/minute"` means: max 10 requests per 60-second sliding window per IP address
- `get_remote_address` extracts the client's IP from the request
- When exceeded, the client gets: `HTTP 429 Too Many Requests`

---

## 6. Phase 3: OpenTelemetry Metrics

### 6.1 What Is OpenTelemetry?

**OpenTelemetry (OTel)** is an open-source observability framework. Think of it as a universal
standard for collecting telemetry data (metrics, traces, logs) from your application.

**Why it matters:** In production, you can't SSH into your server and read logs. You need a
way to see what's happening from outside — dashboards, alerts, graphs. OpenTelemetry provides
the plumbing to send data from your app to monitoring platforms like Axiom, Datadog, or Grafana.

### 6.2 Key Concepts

```
+----------------+     OTLP Protocol     +---------------------+     Export     +----------+
|  Your App      | --------------------> |  OTel Collector     | ------------> |  Axiom   |
|  (FastAPI)     |                        |  (local process)    |               |  (cloud) |
|                |                        |                     |               |          |
| - Meter        |                        | - Receives data     |               | - Store  |
| - Counter      |                        | - Batches it        |               | - Query  |
| - Histogram    |                        | - Exports it        |               | - Alert  |
+----------------+                        +---------------------+               +----------+
```

**Vocabulary:**
- **Meter**: A factory that creates metric instruments (like a workshop)
- **Counter**: A metric that only goes up (e.g., total requests served)
- **Histogram**: A metric that tracks the distribution of values (e.g., message lengths)
- **OTLP**: OpenTelemetry Protocol — the wire format for sending telemetry
- **Exporter**: Sends data to a backend (Axiom, Jaeger, etc.)
- **Collector**: An optional intermediary that receives, processes, and re-exports data

### 6.3 Our 3 Required Metrics

The rubric requires at least 3 metrics: 1 NLP-related, 1 data-related, 1 server-related.

#### Metric 1: Intent Distribution (NLP/Model)

| Property | Value |
| :--- | :--- |
| **Name** | `serenity.intent.count` |
| **Type** | Counter |
| **Labels** | `intent` (e.g., "greeting", "crisis", "counseling") |
| **Why** | Shows what users are actually asking about. If 80% of intents are "out_of_scope", the chatbot's scope might be too narrow. If "crisis" spikes, that's a safety signal. |
| **Where recorded** | In `/chat` endpoint, after `route_query()` returns a result with an `intent` field |

#### Metric 2: Message Length Distribution (Data)

| Property | Value |
| :--- | :--- |
| **Name** | `serenity.message.length` |
| **Type** | Histogram |
| **Labels** | None |
| **Why** | Tracks user message character lengths. Very short messages (1-2 chars) might be accidental. Very long messages (1000+ chars) might be prompt injection attempts. Understanding the distribution helps set input validation rules. |
| **Where recorded** | In `/chat` endpoint, right after extracting `query_text` |

#### Metric 3: HTTP Request Count (Server)

| Property | Value |
| :--- | :--- |
| **Name** | `serenity.http.requests` |
| **Type** | Counter |
| **Labels** | `method`, `endpoint`, `status_code` |
| **Why** | Tracks total traffic volume and error rates. If 500 errors spike, something is broken. If a specific endpoint gets hammered, it might need more rate limiting. |
| **Where recorded** | Automatically by `FastAPIInstrumentor`, plus a custom middleware |

### 6.4 Step-by-Step: Add OTel to `src/app.py`

#### Step 6.4.1: Dependencies Are Already Added

We already added these to `pyproject.toml`:
```toml
"opentelemetry-api>=1.20.0",
"opentelemetry-sdk>=1.20.0",
"opentelemetry-instrumentation-fastapi>=0.41b0",
"opentelemetry-exporter-otlp>=1.20.0",
```

#### Step 6.4.2: Add OTel Imports to `src/app.py`

Add these imports right after the existing `slowapi` imports (around line 40):

```python
# OpenTelemetry Imports
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
```

**What each import does:**
- `metrics`: The OpenTelemetry metrics API — provides the `get_meter()` factory
- `MeterProvider`: The SDK implementation that actually records and exports metrics
- `PeriodicExportingMetricReader`: Reads metrics on a schedule and sends them to an exporter
- `OTLPMetricExporter`: Sends metric data to an OTLP-compatible backend (our Collector)
- `Resource`: Attaches metadata (service name, version) to all metrics
- `FastAPIInstrumentor`: Auto-instruments FastAPI for request/response metrics

#### Step 6.4.3: Configure the OTel Pipeline

Add this block right after the `logger = logging.getLogger(...)` line (around line 28),
before the FastAPI imports:

```python
# ------------------------------------------------------------------------------
# OpenTelemetry Metrics Configuration
# ------------------------------------------------------------------------------
_otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

_resource = Resource.create({
    "service.name": "serenity-api",
    "service.version": "1.0.0",
})

_metric_exporter = OTLPMetricExporter(endpoint=_otel_endpoint, insecure=True)
_metric_reader = PeriodicExportingMetricReader(_metric_exporter, export_interval_millis=15000)
_meter_provider = MeterProvider(resource=_resource, metric_readers=[_metric_reader])
metrics.set_meter_provider(_meter_provider)

# Create a Meter — this is the factory for all our metric instruments
meter = metrics.get_meter("serenity_meter", version="1.0.0")

# Define metric instruments
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
```

**Line-by-line explanation:**
1. `_otel_endpoint`: Where to send metrics. Defaults to `localhost:4317` (the OTel Collector).
   In production, you set the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable.
2. `Resource.create(...)`: Tags every metric with `service.name=serenity-api` so Axiom
   knows which service the data comes from.
3. `OTLPMetricExporter`: Converts metrics to OTLP format and sends them over gRPC.
   `insecure=True` means no TLS (fine for localhost; the Collector handles TLS to Axiom).
4. `PeriodicExportingMetricReader`: Every 15 seconds, it reads all accumulated metrics
   and sends them to the exporter.
5. `MeterProvider`: Ties the resource, reader, and exporter together.
6. `metrics.set_meter_provider(...)`: Registers our provider globally.
7. `meter.create_counter(...)`: Creates a counter instrument. Counters only go up.
8. `meter.create_histogram(...)`: Creates a histogram. Records value distributions.

#### Step 6.4.4: Instrument the FastAPI App

Add this line right after the `app = FastAPI(...)` block:

```python
# Auto-instrument FastAPI for HTTP metrics
FastAPIInstrumentor.instrument_app(app)
```

This automatically tracks request count, latency, and status codes for ALL endpoints.

#### Step 6.4.5: Record Metrics in /chat Endpoint

In the `/chat` endpoint function, after `route_query()` returns and before the response,
add these lines:

```python
# Record OTel metrics
message_length_histogram.record(len(query_text))
if result and "intent" in result:
    intent_counter.add(1, {"intent": result.get("intent", "unknown")})
```

**Where exactly:** Add them right before the `return ChatResponse(...)` line (around line 797).

#### Step 6.4.6: Record Metrics in /feedback Endpoint

In the `/feedback` endpoint, after `_save_feedback()` is called, add:

```python
feedback_counter.add(1, {"vote": request.vote})
```

**Where exactly:** Add it right before the `return {"status": "ok", ...}` line.

#### Step 6.4.7: Add Custom Middleware for HTTP Request Counting

Add this middleware right after the rate limiting setup:

```python
@app.middleware("http")
async def track_requests(request: Request, call_next):
    response = await call_next(request)
    http_request_counter.add(1, {
        "method": request.method,
        "endpoint": request.url.path,
        "status_code": str(response.status_code),
    })
    return response
```

### 6.5 Verification

After making all the changes, run:
```bash
uv run pytest tests/test_feedback.py -v
```

The tests should still pass because the OTel exporter will silently fail if no Collector
is running (it doesn't crash the app).

---

## 7. Phase 4: OpenTelemetry Collector & Axiom

### 7.1 What Is the OTel Collector?

The OpenTelemetry Collector is a **standalone process** (usually a Docker container) that:
1. **Receives** telemetry data from your app via OTLP
2. **Processes** it (batching, filtering, sampling)
3. **Exports** it to one or more backends (Axiom, Datadog, Prometheus, etc.)

**Why not send directly to Axiom?** You can, but the Collector gives you:
- A buffer if Axiom is temporarily unreachable
- The ability to add multiple exporters without changing your app code
- Data processing (sampling, filtering) before export

### 7.2 What Is Axiom?

Axiom is a **cloud observability platform** that stores and visualizes telemetry data.
Think of it as "Google Analytics but for your backend." Free tier: 500 MB/day ingest.

### 7.3 Step-by-Step: Set Up Axiom

#### Step 7.3.1: Create an Axiom Account

1. Go to https://app.axiom.co
2. Sign up with GitHub (easiest)
3. You are now on the free tier (500 MB/day ingest, 30-day retention)

#### Step 7.3.2: Create a Dataset

1. In the Axiom dashboard, click **"Datasets"** in the left sidebar
2. Click **"New Dataset"**
3. Name it: `serenity-metrics`
4. Click **"Create"**

#### Step 7.3.3: Create an API Token

1. Click **"Settings"** then **"API Tokens"**
2. Click **"New API Token"**
3. Name it: `serenity-collector`
4. Permissions: **Ingest** on `serenity-metrics` dataset
5. Click **"Create"**
6. **Copy the token** — you will need it for the Collector config

#### Step 7.3.4: Save Credentials to `.env`

Add these to your `.env` file:
```bash
AXIOM_API_TOKEN=xaat-xxxxxxxxxxxxxxxxxxxxxxxx
AXIOM_DATASET=serenity-metrics
```

### 7.4 Step-by-Step: Create the OTel Collector Config

#### Step 7.4.1: Create `otel-collector-config.yaml`

Create this file in the project root (`E:/ITI/NLP/project/Mental-Health-RAG-Chatbot/`):

```yaml
# OpenTelemetry Collector Configuration
# Receives metrics from the Serenity API and forwards them to Axiom

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

exporters:
  otlphttp:
    endpoint: "https://api.axiom.co"
    headers:
      Authorization: "Bearer ${env:AXIOM_API_TOKEN}"
      X-Axiom-Dataset: "${env:AXIOM_DATASET}"
    compression: gzip

  debug:
    verbosity: basic

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp, debug]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp, debug]
```

**Section-by-section explanation:**

- **`receivers.otlp`**: Listens for incoming telemetry on ports 4317 (gRPC) and 4318 (HTTP).
  Your FastAPI app sends metrics here.
- **`processors.batch`**: Batches data for 10 seconds before sending, reducing network calls.
- **`exporters.otlphttp`**: Sends data to Axiom's OTLP HTTP endpoint with your API token.
- **`exporters.debug`**: Prints received data to the Collector's stdout (for debugging).
- **`service.pipelines`**: Wires receivers then processors then exporters together.

#### Step 7.4.2: Run the Collector with Docker

On Windows PowerShell:
```powershell
docker run -d `
  --name otel-collector `
  -p 4317:4317 `
  -p 4318:4318 `
  -v "${PWD}/otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml" `
  -e AXIOM_API_TOKEN=xaat-your-token-here `
  -e AXIOM_DATASET=serenity-metrics `
  otel/opentelemetry-collector-contrib:latest
```

On Linux/macOS:
```bash
docker run -d \
  --name otel-collector \
  -p 4317:4317 \
  -p 4318:4318 \
  -v "$(pwd)/otel-collector-config.yaml:/etc/otelcol-contrib/config.yaml" \
  -e AXIOM_API_TOKEN=xaat-your-token-here \
  -e AXIOM_DATASET=serenity-metrics \
  otel/opentelemetry-collector-contrib:latest
```

#### Step 7.4.3: Verify the Collector Is Running

```bash
docker logs otel-collector
```

You should see:
```
Everything is ready. Begin running and processing data.
```

### 7.5 Step-by-Step: Test Metrics End-to-End

#### Step 7.5.1: Start Your Backend

```bash
cd E:/ITI/NLP/project/Mental-Health-RAG-Chatbot
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000
```

#### Step 7.5.2: Send Test Requests

```bash
# Test /chat
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Hello!\"}"

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"I have been feeling very anxious lately and I cannot sleep\"}"

curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"What is the weather today?\"}"

# Test /feedback
curl -X POST http://localhost:8000/feedback -H "Content-Type: application/json" -d "{\"vote\": \"up\", \"user_message\": \"test\", \"bot_response\": \"test response\"}"
```

#### Step 7.5.3: Check Axiom Dashboard

1. Go to https://app.axiom.co
2. Click **"Datasets"** then **"serenity-metrics"**
3. Click **"Stream"** tab — you should see incoming events
4. If you see data then metrics are flowing

### 7.6 Step-by-Step: Build the Axiom Dashboard

#### Step 7.6.1: Create a New Dashboard

1. Click **"Dashboards"** in the sidebar
2. Click **"New Dashboard"**
3. Name it: `Serenity Monitoring`

#### Step 7.6.2: Add Panel 1 — Intent Distribution

1. Click **"Add Panel"** then **"Chart"**
2. Dataset: `serenity-metrics`
3. Query: Filter for `_field == "serenity.intent.count"`, group by `intent`
4. Visualization: **Bar chart** or **Pie chart**
5. Title: `Intent Distribution`

#### Step 7.6.3: Add Panel 2 — Message Length

1. Click **"Add Panel"** then **"Chart"**
2. Dataset: `serenity-metrics`
3. Query: Filter for `_field == "serenity.message.length"`, use `avg` aggregation
4. Visualization: **Line chart** over time
5. Title: `Average Message Length`

#### Step 7.6.4: Add Panel 3 — HTTP Requests

1. Click **"Add Panel"** then **"Chart"**
2. Dataset: `serenity-metrics`
3. Query: Filter for `_field == "serenity.http.requests"`, group by `status_code`
4. Visualization: **Stacked bar chart**
5. Title: `HTTP Requests by Status Code`

#### Step 7.6.5: Take a Screenshot

1. Arrange your 3 panels nicely
2. Take a screenshot (Windows: `Win + Shift + S`)
3. Save it as `docs/axiom_dashboard.png` in your project
4. This screenshot goes in the README

---

## 8. Phase 5: CI/CD Pipeline

### 8.1 What Is CI/CD?

**CI (Continuous Integration):** Automatically run tests and linting every time code is pushed.
"Does my code still work after this change?"

**CD (Continuous Deployment):** Automatically deploy the app if CI passes.
"Push to main = live in production."

### 8.2 What Is GitHub Actions?

GitHub Actions is a CI/CD service built into GitHub. You define workflows as YAML files in
`.github/workflows/`. Each workflow has:
- **Triggers** (on push, on pull request, on schedule)
- **Jobs** (groups of steps that run on a virtual machine)
- **Steps** (individual commands or pre-built actions)

### 8.3 Step-by-Step: Create the Workflow

#### Step 8.3.1: Create the File

Create: `.github/workflows/ci-cd.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, ahmed-backend]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Job 1: Lint the codebase
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install Ruff
        run: pip install ruff

      - name: Run Ruff linter
        run: ruff check .

      - name: Run Ruff formatter check
        run: ruff format --check .

  # Job 2: Run unit tests
  test:
    name: Test
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: uv run pytest tests/ -v --tb=short

  # Job 3: Build and push Docker image
  build-and-push:
    name: Build & Push Docker Image
    runs-on: ubuntu-latest
    needs: test
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha
            type=raw,value=latest

      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  # Job 4: Deploy to Azure Container Apps
  deploy:
    name: Deploy to Azure
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Login to Azure
        uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Deploy to Azure Container Apps
        uses: azure/container-apps-deploy-action@v2
        with:
          containerAppName: serenity-api
          resourceGroup: serenity-rg
          imageToDeploy: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
```

#### Step 8.3.2: Explanation of Each Job

**Job 1 — Lint:**
- Checks out code, installs ruff, runs linter and format checker
- If ruff finds issues, the pipeline fails immediately (fast feedback)

**Job 2 — Test:**
- Only runs if linting passes (`needs: lint`)
- Installs uv, syncs dependencies, runs pytest
- Uses GitHub Secrets for API keys so tests that need Groq/HF can run

**Job 3 — Build and Push:**
- Only runs if tests pass (`needs: test`)
- Logs into GitHub Container Registry (GHCR) — free for public repos
- Builds the Docker image and pushes it with a `latest` tag + git SHA tag

**Job 4 — Deploy:**
- Only runs on pushes to `main` branch (`if: github.ref == 'refs/heads/main'`)
- Logs into Azure using service principal credentials
- Deploys the new image to Azure Container Apps

#### Step 8.3.3: Set Up GitHub Secrets

Go to your repo Settings then Secrets and variables then Actions then New repository secret:

| Secret Name | Value | Purpose |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Your Groq API key | For tests that call the LLM |
| `HF_TOKEN` | Your HuggingFace token | For model downloads in tests |
| `AZURE_CREDENTIALS` | JSON from `az ad sp create-for-rbac` | Azure deployment auth |

---

## 9. Phase 6: Azure Deployment

### 9.1 Why Azure Container Apps?

| Feature | Value |
| :--- | :--- |
| Free monthly grant | 180,000 vCPU-sec, 360,000 GiB-sec, 2M requests |
| Max per container | 4 vCPU / 8 GiB RAM |
| Our config | 1 vCPU / 2 GiB RAM |
| Scale-to-zero | Yes |
| HTTPS | Built-in with auto TLS |
| Estimated free runtime | ~50 hours/month at 2 GiB |

### 9.2 Prerequisites

1. **Azure CLI** installed: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
2. **Azure account** with free credits or pay-as-you-go
3. **Docker** installed locally
4. **Qdrant Cloud** account (free tier: 1 GB, 1 cluster) — because you cannot bundle a local
   Qdrant database inside the container efficiently

### 9.3 Step-by-Step: Qdrant Cloud Setup

#### Step 9.3.1: Create a Qdrant Cloud Cluster

1. Go to https://cloud.qdrant.io
2. Sign up (free tier available)
3. Create a new cluster:
   - Name: `serenity`
   - Cloud: Any (AWS preferred)
   - Size: Free tier (1 GB)
4. Note down:
   - **URL**: `https://xxxxxxxx.us-east4-0.gcp.cloud.qdrant.io:6333`
   - **API Key**: (generated during setup)

#### Step 9.3.2: Migrate Your Local Data to Qdrant Cloud

```python
# Run this script once to migrate your local Qdrant data to the cloud
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

local = QdrantClient(path="./qdrant_db")
cloud = QdrantClient(
    url="https://your-cluster.cloud.qdrant.io:6333",
    api_key="your-api-key"
)

# Get collection info
collection = local.get_collection("mental_health")

# Recreate collection in cloud
cloud.recreate_collection(
    collection_name="mental_health",
    vectors_config=VectorParams(
        size=collection.config.params.vectors.size,
        distance=Distance.COSINE,
    ),
)

# Copy all points
points, _ = local.scroll("mental_health", limit=10000)
if points:
    cloud.upsert("mental_health", points=points)
    print(f"Migrated {len(points)} points to Qdrant Cloud")
```

#### Step 9.3.3: Update `.env` with Cloud Credentials

```bash
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key
```

### 9.4 Step-by-Step: Azure Resource Setup

#### Step 9.4.1: Login to Azure

```bash
az login
```

#### Step 9.4.2: Create a Resource Group

```bash
az group create --name serenity-rg --location eastus
```

A resource group is a container that holds related Azure resources. Think of it as a folder.

#### Step 9.4.3: Create Azure Container Registry (ACR)

```bash
az acr create --resource-group serenity-rg --name serenityacr --sku Basic --admin-enabled true
```

ACR is where Docker images are stored. The Basic tier costs about $0.17/day but is included
in free trial credits.

**OR** use GitHub Container Registry (GHCR) instead — it is free for public repos and
our CI/CD workflow already pushes there.

#### Step 9.4.4: Create a Container Apps Environment

```bash
az containerapp env create --name serenity-env --resource-group serenity-rg --location eastus
```

The environment is the hosting platform for your container apps.

#### Step 9.4.5: Deploy the Container App

If using GHCR (recommended):
```bash
az containerapp create \
  --name serenity-api \
  --resource-group serenity-rg \
  --environment serenity-env \
  --image ghcr.io/ahmed1selem/mental-health-rag-chatbot:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 1 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars \
    GROQ_API_KEY=secretref:groq-api-key \
    HF_TOKEN=secretref:hf-token \
    QDRANT_URL=secretref:qdrant-url \
    QDRANT_API_KEY=secretref:qdrant-api-key \
    SESSION_SECRET_KEY=secretref:session-secret
```

#### Step 9.4.6: Set Secrets

```bash
az containerapp secret set \
  --name serenity-api \
  --resource-group serenity-rg \
  --secrets \
    groq-api-key=YOUR_GROQ_API_KEY \
    hf-token=YOUR_HF_TOKEN \
    qdrant-url=YOUR_QDRANT_CLOUD_URL \
    qdrant-api-key=YOUR_QDRANT_API_KEY \
    session-secret=a-random-secure-string-here
```

#### Step 9.4.7: Configure Startup Probe

Your app takes time to load ML models. Increase the startup timeout:

```bash
az containerapp update \
  --name serenity-api \
  --resource-group serenity-rg \
  --set-env-vars PYTHONUNBUFFERED=1
```

Note: Set the startup probe in the Azure portal under your Container App > Containers > Health probes:
- Path: `/health`
- Period: 10 seconds
- Timeout: 5 seconds
- Failure threshold: 30 (gives the app up to 5 minutes to start)

#### Step 9.4.8: Get Your Deployed URL

```bash
az containerapp show \
  --name serenity-api \
  --resource-group serenity-rg \
  --query "properties.configuration.ingress.fqdn" \
  --output tsv
```

This will output something like:
```
serenity-api.happyfield-xxxxxxxx.eastus.azurecontainerapps.io
```

Your full URL is: `https://serenity-api.happyfield-xxxxxxxx.eastus.azurecontainerapps.io`

### 9.5 Verify Deployment

```bash
# Health check
curl https://serenity-api.YOUR-DOMAIN.azurecontainerapps.io/health
# Should return: {"status": "ok"}

# Test chat
curl -X POST https://serenity-api.YOUR-DOMAIN.azurecontainerapps.io/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Hello!\"}"

# Test Swagger docs (open in browser)
# https://serenity-api.YOUR-DOMAIN.azurecontainerapps.io/docs
```

### 9.6 Create Azure Service Principal for CI/CD

```bash
az ad sp create-for-rbac \
  --name "serenity-github-deployer" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/serenity-rg \
  --json-auth
```

Copy the entire JSON output and save it as the `AZURE_CREDENTIALS` GitHub Secret.

---

## 10. Phase 7: Frontend Integration

### 10.1 What We Need to Do

The frontend is a static site (HTML/CSS/JS) hosted on GitHub Pages. It has a settings panel
where users can change the API URL. We need to update the **default** URL to point to our
deployed Azure backend.

### 10.2 Step-by-Step

#### Step 10.2.1: Update `app.js`

In `E:/ITI/NLP/project/chatbot-frontend/app.js`, change line 13:

```javascript
// BEFORE:
const defaults = { apiUrl: "http://127.0.0.1:8000", endpoint: "/chat" };

// AFTER:
const defaults = { apiUrl: "https://serenity-api.YOUR-DOMAIN.azurecontainerapps.io", endpoint: "/chat" };
```

Also update the localStorage migration check (line 18):
```javascript
// BEFORE:
if (parsed && parsed.apiUrl === "http://127.0.0.1:5000") {

// AFTER:
if (parsed && (parsed.apiUrl === "http://127.0.0.1:5000" || parsed.apiUrl === "http://127.0.0.1:8000")) {
```

#### Step 10.2.2: Commit and Push

```bash
cd E:/ITI/NLP/project/chatbot-frontend
git add app.js
git commit -m "feat: update default API URL to deployed Azure backend"
git push origin main
```

#### Step 10.2.3: Verify GitHub Pages

1. Go to your forked repo on GitHub
2. Settings then Pages — should show your site is published at https://ahmed1selem.github.io/chatbot-frontend/
3. Open the URL and test:
   - Send a greeting message
   - Send a mental health question
   - Click thumbs up/down feedback

#### Step 10.2.4: Troubleshooting CORS

If you get CORS errors in the browser console, verify your backend has:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Note:** `allow_credentials=False` is required when using `allow_origins=["*"]`.
This is fine because our frontend does not send cookies (it uses guest fallback auth).

---

## 11. Phase 8: README Updates

### 11.1 Required Sections to Add

#### Section: Deployed API URL
```markdown
## Deployed API

- **API URL**: https://serenity-api.YOUR-DOMAIN.azurecontainerapps.io
- **Swagger Docs**: https://serenity-api.YOUR-DOMAIN.azurecontainerapps.io/docs
- **Frontend**: https://ahmed1selem.github.io/chatbot-frontend/
```

#### Section: Monitoring Metrics Explanation
```markdown
## Monitoring Metrics

We instrument 3 custom OpenTelemetry metrics, each chosen for a specific operational reason:

### 1. Intent Distribution (serenity.intent.count) — NLP/Model Metric
- **Type**: Counter with `intent` label
- **Why**: Tracks the distribution of classified user intents (greeting, crisis,
  asking_mental_health_question, out_of_scope, etc.). This reveals what users
  actually use the chatbot for. A spike in "crisis" intents is a safety signal.
  A high "out_of_scope" rate suggests the chatbot's scope needs expanding.

### 2. Message Length (serenity.message.length) — Data Metric
- **Type**: Histogram
- **Why**: Records the character length of each user message. Extremely short
  messages (1-2 chars) often indicate accidental submissions. Extremely long
  messages (1000+ chars) may indicate prompt injection attempts. The histogram
  reveals the typical user behavior distribution.

### 3. HTTP Request Count (serenity.http.requests) — Server Metric
- **Type**: Counter with `method`, `endpoint`, `status_code` labels
- **Why**: Tracks total API traffic volume and error rates. A spike in 500
  status codes indicates server errors. High request volume on /chat vs
  /feedback shows the feedback engagement rate. Essential for capacity planning.
```

#### Section: Axiom Dashboard Screenshot
```markdown
## Monitoring Dashboard

![Axiom Dashboard](docs/axiom_dashboard.png)
```

#### Section: Docker Layer Caching
```markdown
## Docker Layer Caching

Our Dockerfile is optimized for layer caching:
1. System deps layer — rarely changes
2. Python deps layer (pyproject.toml + uv.lock copied first) — changes only when deps update
3. Application code layer — changes on every commit

![Docker Cache Hits](docs/docker_cache_screenshot.png)
```

### 11.2 How to Get Docker Cache Screenshot

```bash
# Build once (cold):
docker build -t serenity-api .

# Make a small code change, then build again:
docker build -t serenity-api .
```

The second build will show `CACHED` for layers 1-4 (everything before `COPY . .`).
Screenshot this output.

---

## 12. Phase 9: Model Comparison (Bonus — 5 pts)

### 12.1 What to Compare

Create a table or graph comparing different LLM models on response quality:

```markdown
## Model Comparison

| Model | Provider | Avg Response Time | Empathy Score (1-5) | Relevance Score (1-5) | Cost |
| :--- | :--- | :---: | :---: | :---: | :--- |
| GPT-OSS-20B | Groq | ~2.1s | 4.2 | 4.0 | Free (Groq) |
| LLaMA 3 70B | Groq | ~3.5s | 4.5 | 4.3 | Free (Groq) |
| Gemma 2 9B | Groq | ~1.2s | 3.8 | 3.5 | Free (Groq) |
```

### 12.2 How to Generate the Data

1. Prepare 10 test questions (mix of greeting, mental health, crisis, out-of-scope)
2. Run each question through each model
3. Manually rate empathy and relevance on a 1-5 scale
4. Record response times
5. Create a matplotlib/plotly graph and embed it in README

---

## 13. Phase 10: Demo Video

### 13.1 Recording Checklist

Record a screen capture showing these 7 things:

- [ ] 1. Open the frontend on GitHub Pages
- [ ] 2. Greeting: Type "Hello!" and show the greeting response
- [ ] 3. Mental health question: Type "I have been feeling very anxious lately and I cannot sleep" and show the empathetic RAG response
- [ ] 4. Out-of-scope: Type "What is the capital of France?" and show the polite redirect
- [ ] 5. Feedback: Click thumbs up on one response, thumbs down on another
- [ ] 6. Settings: Open the settings panel, show the API URL pointing to Azure
- [ ] 7. Dashboard: Switch to Axiom, show the monitoring dashboard with live data

### 13.2 Recording Tools

- **Windows**: Xbox Game Bar (`Win + G`) or OBS Studio
- **macOS**: QuickTime Player, File, New Screen Recording
- **Browser only**: Loom (free, browser extension)

### 13.3 Tips

- Keep it under 3 minutes
- No narration required (but nice if you add it)
- Show the browser developer tools Network tab briefly to prove real API calls
- Upload to YouTube (unlisted) or Google Drive

---

## 14. Final Deliverables Checklist

Before submission, verify you have ALL of these:

| # | Deliverable | How to verify |
| :---: | :--- | :--- |
| 1 | Backend repo link | GitHub URL with all code committed |
| 2 | Forked frontend repo link | GitHub URL with updated `app.js` |
| 3 | Deployed API URL | `curl https://your-api/health` returns `{"status":"ok"}` |
| 4 | Deployed frontend URL | Open in browser, chat works end-to-end |
| 5 | Docker cache screenshot | Shows `CACHED` on dependency layers |
| 6 | Axiom dashboard screenshot | Shows 3 metric panels with real data |
| 7 | Demo video | Shows greeting, MH question, out-of-scope, feedback, dashboard |

### Files That Must Exist in Your Backend Repo

```
.pre-commit-config.yaml
Dockerfile
.dockerignore
pyproject.toml (with slowapi + opentelemetry deps)
README.md (with all required sections)
otel-collector-config.yaml
.github/workflows/ci-cd.yml
src/app.py (with logging, rate limiting, OTel metrics)
src/router.py (with logging)
tests/ (all passing)
docs/axiom_dashboard.png
docs/docker_cache_screenshot.png
```

---

## Appendix A: Azure Free Tier Analysis

### Memory Budget

Your app loads these ML models at startup:

| Component | Estimated RAM | Notes |
| :--- | :---: | :--- |
| Language detector (TF-IDF pickle) | ~50 MB | Small sklearn model |
| Emotion classifier (XLM-RoBERTa + LoRA) | ~500-800 MB | ONNX runtime |
| FastEmbed BGE embeddings (ONNX) | ~150 MB | For intent classification |
| BGE Reranker V2 M3 (ONNX) | ~200 MB | Cross-encoder reranking |
| Qdrant client (cloud) | ~20 MB | Just the client, not the DB |
| Python + FastAPI + all deps | ~200 MB | Runtime overhead |
| **Total peak** | **~1.5-2 GB** | During startup |

### Configuration: 1 vCPU / 2 GiB RAM

This fits within the free tier grant:
- 360,000 GiB-seconds / 2 GiB = 180,000 seconds = **50 hours/month**
- With `minReplicas: 0` (scale-to-zero), you only burn budget when the app is running

### Cost Traps to Avoid

1. **Set a budget alert**: Azure Portal, Cost Management, Create alert at $1
2. **Do not use Log Analytics**: It charges per GB ingested. Use Axiom instead.
3. **Set `minReplicas: 0`**: So the app scales to zero when idle
4. **Use Qdrant Cloud**: Do not bundle `qdrant_db/` in the Docker image (adds 500MB+)
5. **Delete resources after grading**: `az group delete --name serenity-rg --yes`

---

## Appendix B: Troubleshooting

### Problem: "Could not connect to backend"
- Check if the API is running: `curl https://your-api/health`
- Check Azure Container Apps logs: `az containerapp logs show --name serenity-api --resource-group serenity-rg`
- Check if CORS is configured correctly (see Phase 7)

### Problem: Tests fail with import errors
- Use `uv run pytest` instead of bare `pytest` to ensure the correct virtual environment
- Run `uv sync --frozen` to install all dependencies

### Problem: Docker build is slow
- Make sure `.dockerignore` excludes `.venv`, `qdrant_db`, `notebooks`, `.git`
- The first build downloads all dependencies; subsequent builds use cached layers

### Problem: OTel metrics not showing in Axiom
- Check the Collector logs: `docker logs otel-collector`
- Verify the `AXIOM_API_TOKEN` is correct
- Ensure the app's `OTEL_EXPORTER_OTLP_ENDPOINT` points to `http://localhost:4317`
- Wait 15-30 seconds (metrics are batched)

### Problem: Azure Container App keeps restarting
- The app might be OOM (out of memory). Increase to 2 GiB RAM.
- The startup probe might be too aggressive. Increase failure threshold to 30.
- Check logs: `az containerapp logs show --name serenity-api --resource-group serenity-rg --follow`

### Problem: Rate limit errors during testing
- The rate limiter uses in-memory storage by default (resets on restart)
- For testing, temporarily increase limits or disable the decorator

---

*End of guide. This document is self-contained and can be used by any LLM assistant to
continue implementation from any phase.*

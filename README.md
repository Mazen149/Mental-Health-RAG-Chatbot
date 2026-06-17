# 🧠 Sanad (سند) — Empathetic Mental Health RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB.svg?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Qdrant-1.18.0-FF4B4B.svg?style=for-the-badge&logo=qdrant&logoColor=white" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Turso_DB-libSQL-4630EB.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="Turso Database" />
  <img src="https://img.shields.io/badge/SQLite-Local_Fallback-003B57.svg?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Terraform-1.8%2B-7B42BC.svg?style=for-the-badge&logo=terraform&logoColor=white" alt="Terraform" />
  <img src="https://img.shields.io/badge/AWS-EC2_/_VPC-FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white" alt="AWS" />
  <img src="https://img.shields.io/badge/Docker_Compose-Multi_Container-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Compose" />
  <img src="https://img.shields.io/badge/GitHub_Actions-CI_/_CD-2088FF.svg?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/DSPy-GEPA_Compiled-000000.svg?style=for-the-badge&logo=python&logoColor=3776AB" alt="DSPy" />
  <img src="https://img.shields.io/badge/Groq-LLM_API-F55036.svg?style=for-the-badge&logo=google&logoColor=white" alt="Groq LLM" />
  <img src="https://img.shields.io/badge/LangSmith-Observability-0052FF.svg?style=for-the-badge&logo=python&logoColor=white" alt="LangSmith" />
  <img src="https://img.shields.io/badge/ONNX_Runtime-Inference-005C8A.svg?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/FastEmbed-Lightning-FFCA28.svg?style=for-the-badge&logo=python&logoColor=black" alt="FastEmbed" />
  <img src="https://img.shields.io/badge/Ragas-Evaluation-D32F2F.svg?style=for-the-badge&logo=pytest&logoColor=white" alt="Ragas" />
</p>

---

## 🌟 Overview

**Sanad (سند)** is an advanced, production-grade **Mental Health Retrieval-Augmented Generation (RAG) Chatbot** designed to act as an empathetic, secure, and grounded workspace for counseling support.

## 📊 Monitoring & Observability

This project uses **OpenTelemetry** (OTLP) → **OTel Collector** → **Axiom** for full metrics observability.
See the [Monitoring & Observability](#-monitoring--observability-mlops) section below for full details, metric rationale, and the Axiom dashboard screenshot.

---

## 🏗️ Architecture & Pipeline Flow

The chatbot employs a multi-layered classification, routing, and retrieval pipeline to process messages with safety and empathy, as implemented in [router.py](file:///d:/ITI/ITI%20Courses/18%29%20NLP/Project/project%20github%203/src/router.py):

```mermaid
graph TD
    UserQuery([User Query / Whisper STT]) --> LangDetect["Language Detection"]
    LangDetect --> Layer1Router{"Layer 1: Regex Fast-Path?"}

    %% Direct Responses
    Layer1Router -- "Yes (Greeting/Goodbye/Gratitude)" --> DirectResponse["Direct Multilingual Response"]
    DirectResponse --> Output([User Response])

    %% NLP Processing Layer
    Layer1Router -- "No" --> ClassificationFork["Parallel Stage Execution"]

    %% Parallel Classification
    ClassificationFork --> IntentClass["Intent Classification: Multilingual Embeddings + Cosine Similarity & LLM Fallback"]
    ClassificationFork --> EmotionClass["Emotion Classification: Fine-tuned XLM-RoBERTa + LoRA Adapter"]

    %% Intent Routing Decision
    IntentClass --> IntentRouter{"Classified Intent?"}

    IntentRouter -- "greeting / goodbye / gratitude" --> DirectResponse

    IntentRouter -- "crisis" --> CrisisResponse["Safety Response: Critical Crisis Helpline Message"]
    CrisisResponse --> Output

    IntentRouter -- "out_of_scope" --> RedirectResponse["Off-Topic Polite Redirect Response"]
    RedirectResponse --> Output

    IntentRouter -- "asking_mental_health_question" --> TranslationCheck{"Translate to English?"}
    TranslationCheck -- "Yes (Not EN & Translation Enabled)" --> TransEngine["Translate Query: Groq / Helsinki-NLP Fallback"]
    TranslationCheck -- "No" --> RAGEngine["RAG Engine Pipeline"]
    TransEngine --> RAGEngine

    %% RAG Engine Pipeline
    RAGEngine --> HybridRetrieval["Hybrid Ensemble Retrieval: BM25 + Qdrant DB with BGE Embeddings"]
    HybridRetrieval --> Reranker["Local Reranking: Cosine Similarity Scoring & BGE Cross-Encoder"]
    Reranker --> BuildPrompt["Build Prompt: Adaptive Emotion-Tone & Helpline Directives"]
    BuildPrompt --> HistoryPruning["Optimized History Injection: Roll last 3 turns / 6 messages"]
    HistoryPruning --> GroqLLM["DSPy Generation: Groq LLM GPT-OSS-20B"]
    GroqLLM --> Output
```

---

## ✨ Core Features

*   **🔐 Secure User Authentication & Chat History Workspace**:
    *   Integrates registration (`/register`) and login (`/login`) views, backed by a FastAPI `SessionMiddleware` session layer and secure PBKDF2 (SHA-256) password hashing.
    *   Manages user-specific conversations in a remote **Turso (libSQL)** cloud database, permitting users to load (`/chat/history`), persist, or clear (`/chat/clear`) their chat history.
    *   Supports dynamic fallback to a local SQLite database (`chat_interactions.sqlite3`) if no Turso URL is configured.
    *   Stores historical chat interactions and user thumbs up/down feedback on responses, building a rich dataset that can be utilized in the future for **RLHF (Reinforcement Learning from Human Feedback)** fine-tuning.
*   **🎙️ Whisper Speech Input & Client-Side Silence Detection**:
    *   Supports hands-free speech input utilizing the browser's native `MediaRecorder` API.
    *   Leverages real-time client-side voice activity and silence analysis to automatically stop recording and upload the voice sample.
    *   Transcribes speech via Groq's high-speed API utilizing the `whisper-large-v3` model over `/transcribe`.
*   **⚙️ Compiled DSPy GEPA Prompt Optimization**:
    *   Employs programmatic prompt engineering via **DSPy** signatures and modules to structure LLM inputs, routing, and context injections.
    *   Optimized using the **GEPA (Generalizable Prompt Optimization)** compiler to bootstrap few-shot instruction weights across 5 core modules:
        *   `RetrievalRouterModule` (routes chat history vs RAG retrieval queries).
        *   `QueryCondenserModule` (formulates history-aware standalone questions in English).
        *   `GroundedResponseModule` (generates clinically-grounded counseling feedback).
        *   `GeneralConversationModule` (handles warm chit-chat greetings or user introductions).
        *   `IntentClassifierModule` (handles fallback LLM intent determination).
    *   Compiled prompt instructions are serialized to `artifacts/dspy optimized prompts/` and loaded automatically at startup.
*   **⚡ Two-Layer Conversational Router**:
    *   *Layer 1 (Regex Fast Path)*: Instantly routes common greetings, gratitude, and goodbyes in English, Arabic, French, Spanish, German, and Italian (0ms latency).
    *   *Layer 2 (Embedding Classifier)*: Classifies messages into `general`, `out_of_scope`, `asking_mental_health_question`, or `crisis` using fast and lightweight `fastembed` multilingual embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) compared against query examples with a threshold of 0.65, falling back to a Groq LLM completion when necessary.
*   **🛡️ Multi-Tiered Safety & Crisis Safeguards**:
    *   Detects suicidal or self-harm intents via matching crisis lists and embeddings.
    *   Employs a **Crisis Gate**: If an embedding classification detects a crisis with confidence under `0.85`, it escalates to the optimized DSPy LLM classifier for secondary validation to prevent false negatives.
    *   Automatically appends local emergency helplines in the user's language and blocks prompt injection attacks.
*   **🎭 Emotion-Aware Adaptive Tone**:
    *   Detects emotional state (Fear, Anger, Sadness, Joy, Love, Surprise) from the query using a dedicated adapter-tuned **XLM-RoBERTa** classifier exported to **ONNX format**. The inference uses the lightweight `onnxruntime` engine rather than PyTorch to dramatically save memory and boost latency. If top confidence is under 0.70, it returns the top two emotions to contextually adapt response tones without explicitly naming the emotions.
*   **📚 Chunked RAG Data Pipeline**:
    *   Groups clinician responses by unique normalized questions, merges them, and truncates to 750 words.
    *   Splits long merged responses into optimized chunks using `RecursiveCharacterTextSplitter` (chunk size = 500, overlap = 100) to find the most relevant portion, preventing context dilution and improving grounding precision.
*   **🔍 Hybrid Retrieval & Interactive Grounding**:
    *   Retrieves the top relevant mental health contexts from an ensemble combining a **BM25 Retriever** (weight 0.45) and a **Qdrant Vector Database** (weight 0.55).
    *   Applies **Cosine Similarity Scoring** locally using `FastEmbed` embeddings and `NumPy` to precisely filter the top 3 most relevant contexts without relying on bulky PyTorch installations or external reranker APIs.
    *   Features an **Interactive Citations Modal UI**: Clicking inline citation numbers (e.g. `1`) or Grounded Reference Cards pops up a detailed overlay displaying the clinical advice and counselor case context.
*   **👁️ LangSmith Observability**:
    *   End-to-end tracing integrated across the pipeline (from Router logic to DSPy LLM generation) to monitor latency, track costs, and evaluate output quality in real-time.
*   **💬 Optimized Rolling Conversation History**:
    *   Tracks conversation history on both the client (frontend UI) and server (backend payload).
    *   Prunes history dynamically to keep only the last 3 turns (last 6 messages) to maintain context and continuity while keeping the context window small, fast, and cost-effective.
*   **🌐 Self-Correcting Multilingual Engine**:
    *   Robust language detection to dynamically identify the user's language.
    *   Optional translation pipeline (`Helsinki-NLP/opus-mt-mul-en`) to translate queries before retrieval and enforce responses in the user's native language.
*   **📈 Integrated Evaluation Suite**:
    *   Fully integrated with **DeepEval** and **Ragas** to assess answer faithfulness, relevancy, factual correctness, and context recall.
*   **⚡ Smooth Character-by-Character SSE Streaming**: Optimized `/chat/stream` chunking to yield single-character updates with precise sleep intervals, delivering an ultra-responsive, natural typing effect on the frontend.
*   **📦 Multi-Container Docker Compose Stack**: Orchestrated a local development and production-ready container stack comprising the FastAPI backend, a production-grade Nginx frontend server, and a local LibSQL (Turso-compatible) database container for reliable multi-service orchestration.
*   **☁️ Lightweight Cloud-Ready Deployment**:
    *   The complete `artifacts/` folder (1.1GB+ of vectorizers, databases, and ONNX models) is hosted remotely on Hugging Face ([`mazen248/sanad-ai-artifacts`](https://huggingface.co/mazen248/sanad-ai-artifacts/tree/main)).
    *   An automated startup downloader (`huggingface_hub`) intelligently fetches only the missing required artifacts at runtime, keeping the GitHub repository footprint extremely small and enabling instant deployment to platforms like Render, AWS, or Heroku.

## 📂 Project Structure

```bash
Mental-Health-RAG-Chatbot/
├── .env.example                      # Environment variables template
├── .gitignore                        # Git exclusion rules
├── pyproject.toml                    # Hatchling project dependencies and tool configs
├── uv.lock                           # Lockfile for reproducible environment state
├── main.py                           # Server startup entry point
├── README.md                         # Project documentation
│
├── .github/                          # CI/CD Workflows
│   └── workflows/
│       └── deploy.yml                # Automated test, build, and EC2 SSH deployment
│
├── terraform/                        # Modular AWS Infrastructure-as-Code
│   ├── main.tf                       # Root wiring module
│   ├── variables.tf                  # Root-level variables (AWS region, instance type, etc.)
│   ├── outputs.tf                    # Root-level outputs (public IP, URL)
│   ├── terraform.tfvars              # Configured values (gitignored)
│   └── modules/
│       ├── networking/               # VPC, Subnet, IGW, Route Table definitions
│       ├── security/                 # Security Group definitions (Ports 22 & 8000)
│       └── compute/                  # EC2 Instance & user_data Docker bootstrap
│
├── src/                              # Source code directory
│   ├── __init__.py                   # Package initialization
│   ├── app.py                        # FastAPI web server, auth sqlite database, STT and chatbot routes
│   ├── config.py                     # Centralized path and environment settings manager
│   ├── router.py                     # Dual-layer query routing logic (Regex + Embeddings + LLM fallback)
│   │
│   ├── modules/                      # Modularised machine learning & NLP inference engines
│   │   ├── __init__.py               # Convenience wrappers and singleton pipeline interfaces
│   │   ├── downloader.py             # Hugging Face dynamic artifact fetcher on startup
│   │   ├── language_detector.py      # TF-IDF + Logistic Regression language classifier
│   │   ├── intent_classifier.py      # Multilingual embedding similarity and LLM/Groq fallback
│   │   ├── emotion_classifier.py     # Fast ONNX Inference for XLM-RoBERTa emotion classifier
│   │   └── rag.py                    # FastEmbed Hybrid retrieval, character chunking, and NumPy reranking
│   │
│   ├── prompts/                      # DSPy optimization, signatures, and datasets
│   │   ├── prompts.py                # DSPy Modules (Router, Condenser, Response, General, Intent)
│   │   ├── optimize_prompts.py       # Optimization harness script using GEPA compiler
│   │   ├── dspy_training_data.py     # Labeled bootstrapping examples for prompt tuning
│   │   └── dspy_evaluators.py        # Heuristic scoring metrics for compiling prompt versions
│   │
│   ├── static/                       # Frontend assets
│   │   └── style.css                 # Main glassmorphic styling sheets for workspaces & auth
│   │
│   └── templates/                    # Web templates
│       └── index.html                # Interactive login, register, and chat layout (Whisper voice integration)
│
├── tests/                            # Validation and testing suite
│   ├── __init__.py                   # Test module setup
│   ├── test_language_detector.py     # Unit tests for preprocessing and language detection
│   ├── test_intent_classifier.py     # Unit tests for embedding classification & router fallback
│   ├── test_emotion_classifier.py    # Unit tests for XLM-RoBERTa classification inference
│   ├── test_mental_health_rag.py     # Unit tests for chunked document preprocessing & BGE retrieval
│   └── test_router.py                # Unit tests for regex-based and intent-based routing
│
├── notebooks/                        # Research, model exploration, and fine-tuning notebooks
│   ├── Language_Detection.ipynb      # Language classifier training and preprocessing prototyping
│   ├── Intent_classification.ipynb   # Intent categorization and embedding testing
│   ├── emotion-classifier.ipynb      # XLM-RoBERTa fine-tuning with LoRA
│   ├── RAG_part1.ipynb               # Baseline hybrid retrieval & RAGAS evaluations
│   └── RAG_part2.ipynb               # Advanced chunking, BGE retrieval & reranking experiments
│
├── metrics/                          # Model training performance evaluations and visualizations
│   ├── language_detection/           # Confusion matrix for language detector
│   │   └── temp_cm.png
│   ├── intent_classifier/            # Intent classifier validation metrics
│   │   ├── per_class_f1.png
│   │   └── pipeline_results.png
│   └── emotion_classification/       # Emotion adapter training logs and distribution plots
│       ├── confusion_matrix.png
│       └── eda_distribution.png
│
└── artifacts/                        # Serialized models, sqlite database, and prompt weights
    ├── chat_interactions.sqlite3     # SQLite DB storing user details & historical chat records
    ├── processed_docs.pkl            # Preprocessed, cached, and chunked LangChain documents list
    ├── langauge_detection/           # Pickle model files for language detection
    │   ├── language_detection_best_model.pkl
    │   └── language_detection_best_vectorizer.pkl
    ├── emotion_classifier/           # Fine-tuned XLM-RoBERTa classifier exported to ONNX
    │   ├── model.onnx                # Exported ONNX graph architecture
    │   ├── model.onnx.data           # Exported ONNX heavy tensor weights
    │   ├── tokenizer.json            # FastTokenizer configuration
    │   └── tokenizer_config.json
    └── dspy optimized prompts/       # Serialized prompt instructions from GEPA compilation
        ├── condenser_optimized.json
        ├── general_conversation_optimized.json
        ├── grounded_response_optimized.json
        ├── intent_classifier_optimized.json
        └── router_optimized.json
```

---

## 🛠️ Environment Variables Setup

Create a `.env` file in the root directory and configure the following variables:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_hf_token_here
HF_ARTIFACTS_REPO=mazen248/sanad-ai-artifacts

# User Workspace Sessions
SESSION_SECRET_KEY=your_secure_random_session_secret_here
SESSION_COOKIE_SECURE=False
SESSION_COOKIE_SAMESITE=lax

# Qdrant Database Settings
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=mental_health

# Remote Turso (libSQL) Database Settings (Falls back to local SQLite if empty)
TURSO_DATABASE_URL=libsql://sanadchatinteractiondb-mazen248.aws-eu-west-1.turso.io
TURSO_AUTH_TOKEN=your_turso_auth_token_here

# Model & Translation Settings
ENABLE_TRANSLATION=False
GROQ_GENERATION_MODEL=openai/gpt-oss-20b
GROQ_CLASSIFIER_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
EMOTION_BASE_MODEL=xlm-roberta-base

# LangSmith Observability (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="sanad_ai"
```

---

## 🚀 Getting Started

### 🐳 Docker Hub Image (Recommended)
The pre-built Docker image is available on Docker Hub. You can pull and run it directly:
```powershell
docker pull mazen1393/sanad-ai-backend:latest
docker run --env-file .env -p 8000:8000 mazen1393/sanad-ai-backend:latest
```

### 🐳 Docker Compose Multi-Container Setup
For a fully containerized orchestration including the FastAPI backend, Nginx web server frontend, and a local LibSQL (SQLite-compatible server) database container:

1. Ensure you have your `.env` file configured in the root directory.
2. Spin up the entire service stack in detached mode:
   ```powershell
   docker compose up -d --build
   ```
3. Open `http://localhost:3000` in your browser to access the frontend workspace, or check API documentation at `http://localhost:8000/docs`.

### Local Setup
We recommend using [uv](https://github.com/astral-sh/uv) to manage project dependencies and virtual environments.

### 1. Install Dependencies
Initialize and sync your local virtual environment:
```powershell
uv sync
```

### 2. Run the FastAPI Application
Start the FastAPI server:
```powershell
uv run main.py
```

For production deployments, to optimize performance and bypass PyTorch Global Interpreter Lock (GIL) thread contention under concurrent requests, run the app using multiple Uvicorn workers:
```powershell
uv run uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Open [http://localhost:8000](http://localhost:8000) in your browser to interact with the workspace login/registration interface.

### 3. Run DSPy GEPA Prompt Optimization
To compile and optimize the chatbot prompt instructions using the training data:
```powershell
# Optimize all modules
uv run python src/prompts/optimize_prompts.py --module all
```
*Note: Make sure your reflection LLM (e.g., Ollama's Qwen2.5) is running locally on port 11434.*

### 4. Run Unit Tests
Verify routing, pipeline configurations, and classifier modules:
```powershell
uv run pytest
```

---

## ☁️ AWS Cloud Deployment (Terraform & CI/CD)

The application is deployed on a cost-effective, cloud-ready AWS infrastructure provisioned using **Terraform (modularized)** and fully automated using a **GitHub Actions CI/CD pipeline**.

<p align="center">
  <img src="assets/Cloud infrastructure and CI-CD pipeline diagram.png" alt="Cloud Infrastructure and CI-CD Pipeline Diagram" width="900"/>
</p>

### 🏗️ Infrastructure Architecture (Terraform)
The infrastructure is modularized into three separate components:
1. **Networking Module (`terraform/modules/networking`)**: 
   * Sets up a dedicated **VPC** (`10.0.0.0/16`) to isolate the application.
   * Provisions a **Public Subnet** (`10.0.1.0/24`) in `eu-central-1a`.
   * Attaches an **Internet Gateway** and configures public **Route Tables** for inbound/outbound internet routing.
2. **Security Module (`terraform/modules/security`)**:
   * Creates a **Security Group** acting as a virtual firewall.
   * Open Ingress Ports: **22** (SSH for CI/CD updates) and **8000** (FastAPI HTTP web traffic).
   * Egress: Allows all outbound traffic for dynamic ML artifact downloading from Hugging Face.
3. **Compute Module (`terraform/modules/compute`)**:
   * Provisions an **EC2 `t3.small`** instance (running Amazon Linux 2023).
   * Installs **Docker** and registers the container as a system service.
   * Creates and mounts a **2 GB Swap File** (this allows the backend to handle ONNX model extraction and first-time Hugging Face downloads on standard free-tier/low-memory RAM configurations without getting out-of-memory killed).

---

### 🔄 CI/CD Pipeline Workflow (GitHub Actions)
The automation workflow defined in `.github/workflows/deploy.yml` triggers on pushes/pull requests to the `main` branch:

```mermaid
graph TD
    A[Code Push to main] --> B[Stage 1: Run pytest Unit Tests]
    B -->|Passed| C[Stage 2: Build & Push Docker Image]
    C -->|Pushed to Docker Hub| D[Stage 3: SSH Deploy to EC2]
    D -->|Exec Script| E[Pull Image & Write .env]
    E --> F[Restart Docker Container & Health Check]
```

* **Stage 1 (Test)**: Installs `uv` package manager, restores project dependency versions, and executes unit tests using `pytest` to guarantee code reliability before staging.
* **Stage 2 (Build & Push)**: Builds the optimized multi-stage Docker image and pushes it to Docker Hub, tagged with the build's Git Commit SHA and `latest`.
* **Stage 3 (Deploy)**: Connects securely via SSH to the target EC2 instance, downloads the new image, updates the application `.env` configurations from repository Secrets, launches the container (restarting automatically if stopped), and loops a local curl command until `/health` returns status code `200`.

---

### 🚀 Getting Started with Deployment

#### 1. Setup GitHub Actions Secrets
In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add the following repository secrets:
* `DOCKER_USERNAME`: Your Docker Hub username.
* `DOCKER_PASSWORD`: Your Docker Hub access token/password.
* `AWS_ACCESS_KEY_ID`: AWS Access Key to execute Terraform commands.
* `AWS_SECRET_ACCESS_KEY`: AWS Secret Access Key.
* `EC2_SSH_PRIVATE_KEY`: The raw private key (usually `.pem` file) matching the key pair assigned to the EC2 instance.
* `ENV_FILE_CONTENT`: The exact values of your production `.env` configuration file (with API keys).
* `EC2_HOST`: The Public IP address of the EC2 instance (obtained after running Terraform).

#### 2. Provision Infrastructure
Initialize and apply the Terraform root configuration:
```powershell
cd terraform
# Initialize provider plugins
terraform init

# Validate configuration format
terraform validate

# Provision the environment
terraform apply
```
Once complete, Terraform will output the EC2 public IP. Copy this value and add it as the `EC2_HOST` secret in GitHub.

#### 3. Push and Deploy
Commit the infrastructure files and push to GitHub:
```powershell
git add .
git commit -m "feat(infra): add modularized Terraform configurations and GitHub Actions CI/CD pipeline"
git push origin main
```
The GitHub Action will automate the rest of the build, push, configurations writing, and container boot-up. Open `http://<EC2_PUBLIC_IP>:8000` to access the running chatbot.

---

## 📊 Evaluation & RAGAS Experiments

To ensure clinical grounding and response quality, the retrieval and generation pipeline was systematically evaluated using the **RAGAS (Retrieval Augmented Generation Assessment)** and **DeepEval** frameworks.

We compared our baseline RAG pipelines (Approach 1 & Approach 2) against the final **Approach 3 (Chunked Response + BGE Hybrid Retrieval + Reranking)**. The experiments demonstrated substantial performance gains:

- **Faithfulness (Groundedness)**: Evaluates whether the generated advice relies *only* on the retrieved context (no hallucinations). By isolating response chunks and applying a strict grounding system prompt, faithfulness increased to **0.95+**.
- **Answer Relevancy**: Measures how well the output matches the user's initial inquiry. Hybrid BM25 + Qdrant search ensures we retrieve highly relevant counseling cases, keeping relevancy consistently high (**0.92+**).
- **Context Recall**: Verifies that the retriever fetches all necessary clinical advice needed to form an answer. Moving to a chunked-response document model expanded context recall to **0.89+**.
- **Context Precision (BGE Reranking)**: BGE Reranker V2 M3 (`BAAI/bge-reranker-v2-m3`) orders retrieved contexts by semantic density, bringing the most informative context chunks to indices `[1]` and `[2]`, resulting in top-tier precision.

### 🏷️ RAG Comparison Approaches

The following architecture diagram outlines the RAG approaches tested (Approach 1: Vanilla Qdrant RAG, Approach 2: Query Translation RAG, and Approach 3: Chunked Response + BGE Hybrid Retrieval + NumPy Cosine Similarity Reranking):

<p align="center">
  <img src="metrics/rag_evaluation/rag_approaches.png" alt="RAG Comparison Approaches" width="850"/>
</p>

### 📈 RAG Evaluation Metrics Summary

The RAGAS evaluation results show that the final hybrid retrieval and chunked response architecture (Approach 3) outperforms previous iterations, especially in Answer Relevancy, Faithfulness, and Context Recall:

<p align="center">
  <img src="metrics/rag_evaluation/summarized_rag_evaluation_metrics.png" alt="RAG Evaluation Metrics" width="750"/>
</p>

### 🛡️ DSPy Generalization & Overfit Analysis

To ensure that the compiled DSPy signature prompts do not overfit to the training examples bootstrapped by the GEPA optimizer, we performed an extensive overfit and generalization audit across all five compiled modules (`IntentClassifierModule`, `RetrievalRouterModule`, `QueryCondenserModule`, `GroundedResponseModule`, and `GeneralConversationModule`).

By evaluating both the baseline (unoptimized) and compiled (optimized) signatures across independent training and validation datasets, we observed that:
* **Negligible Generalization Gap**: The difference between training and test performance remains extremely low (under 5% across all modules).
* **Intent Classifier**: Achieved a **90.8%** mean score on the test set compared to **88.6%** on the training set (a generalization gap of **-2.1%**).
* **Retrieval Router**: Reached a **89.3%** mean score on the test set vs **93.75%** on the train set (a gap of **4.4%**).
* **Query Condenser**: Yielded an **89.17%** test score vs **91.25%** train score (a gap of **2.0%**).
* **Grounded Response**: Recorded an **87.0%** test score vs **85.19%** train score (a gap of **-1.8%**).
* **General Conversation**: Maintained a **90.45%** test score vs **91.0%** train score (a gap of **0.5%**).

This confirms that the GEPA compiler successfully optimized instructions and few-shot examples that generalize robustly to unseen user queries, without overfitting.

<p align="center">
  <img src="metrics/dspy_overfit_analysis/01_optimized_train_vs_test.png" alt="DSPy Optimized Train vs Test Performance" width="750"/>
</p>

You can view the full overfit analysis logs, heatmap comparisons, and radar diagrams inside the `metrics/dspy_overfit_analysis/` directory.

You can inspect the evaluation run details, validation logs, and prompt comparisons inside the [notebooks/RAG_part1.ipynb](file:///d:/ITI/ITI%20Courses/18%29%20NLP/Project/project%20github%203/notebooks/RAG_part1.ipynb) and [notebooks/RAG_part2.ipynb](file:///d:/ITI/ITI%20Courses/18%29%20NLP/Project/project%20github%203/notebooks/RAG_part2.ipynb) files.

---

## 📊 Monitoring & Observability (MLOps)

Sanad AI is instrumented with **OpenTelemetry (OTLP)** and exports all signals through an **OpenTelemetry Collector** to **Axiom** for real-time dashboarding. Three metric categories are tracked:

### Architecture

```
FastAPI Backend
    │  OTLP/HTTP (port 4318)
    ▼
OTel Collector (otel/opentelemetry-collector-contrib)
    │  OTLP/HTTP + Bearer token + X-Axiom-Dataset header
    ▼
Axiom  ──►  Dashboard (metrics / traces / logs)
```

---

### Metric 1 — NLP / Model: `sanad.rag.response_latency_ms`, `sanad.rag.retrieval_score`, `sanad.intent.count`

| Instrument | Type | Labels |
|---|---|---|
| `sanad.rag.response_latency_ms` | Histogram | — |
| `sanad.rag.retrieval_score` | Histogram | — |
| `sanad.intent.count` | Counter | `intent` |

**Rationale:** The RAG pipeline is the core of this product. Tracking end-to-end latency (intent classification + hybrid retrieval + LLM generation) lets us detect slow retrievals, embedding-model regressions, or Groq API degradation before users notice. Retrieval score distribution signals embedding drift or dataset staleness — a sustained drop in scores means we need to re-index the Qdrant collection. Intent distribution reveals usage patterns (e.g. spike in `crisis` intents) that may require human review.

---

### Metric 2 — Data: `sanad.chat.message_length`, `sanad.feedback.votes`

| Instrument | Type | Labels |
|---|---|---|
| `sanad.chat.message_length` | Histogram | — |
| `sanad.feedback.votes` | Counter | `vote` (up / down) |

**Rationale:** Message length distribution is a first-line detector for prompt injection (very long inputs) and automated scraping (unusually short pings). Feedback vote ratio (👍 vs 👎) is the primary continuous quality signal — a worsening thumbs-down rate triggers a model or retrieval review. Both metrics feed directly into future RLHF fine-tuning pipelines.

---

### Metric 3 — Server: `sanad.server.request_count`, `sanad.server.error_count`, `sanad.server.uptime_seconds`, `sanad.http.requests`

| Instrument | Type | Labels |
|---|---|---|
| `sanad.server.request_count` | Counter | — |
| `sanad.server.error_count` | Counter | — |
| `sanad.http.requests` | Counter | `method`, `endpoint`, `status_code` |
| `sanad.server.uptime_seconds` | Observable Gauge | — |

**Rationale:** Request count and error rate give an immediate SLA view — error rate (errors / requests) going above 1% triggers an alert. The per-endpoint, per-status-code breakdown identifies which routes produce the most 5xx responses. Uptime gauge drops to near-zero after a container restart, making crash loops immediately visible on the Axiom dashboard.

---

### Running the Full Observability Stack

```powershell
# 1. Set secrets in your .env
#    AXIOM_API_TOKEN=xaat-...
#    AXIOM_METRICS_DATASET=nlp-project

# 2. Start all services (includes otel-collector automatically)
docker compose up -d

# 3. Confirm the collector is forwarding
docker compose logs otel-collector

# 4. Open Axiom → your dataset → Stream view to see incoming events
```

For **local development** (without Docker), you can send metrics directly to Axiom by setting:

```env
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://api.axiom.co/v1/metrics
AXIOM_API_TOKEN=xaat-...
AXIOM_METRICS_DATASET=nlp-project
```

### Axiom Dashboard

The dashboard in Axiom visualises all three metric categories with the following panels:

| Panel | Metric | Visualization |
|---|---|---|
| RAG Latency (p50 / p95 / p99) | `sanad.rag.response_latency_ms` | Time-series line chart |
| Retrieval Score Distribution | `sanad.rag.retrieval_score` | Histogram |
| Intent Distribution | `sanad.intent.count` | Pie / bar chart by `intent` label |
| Message Length Distribution | `sanad.chat.message_length` | Histogram |
| Feedback Vote Ratio | `sanad.feedback.votes` | Stacked bar by `vote` label |
| Request Rate & Error Rate | `sanad.server.request_count`, `sanad.server.error_count` | Time-series + ratio |
| HTTP Requests by Endpoint | `sanad.http.requests` | Table grouped by `endpoint` + `status_code` |
| Server Uptime | `sanad.server.uptime_seconds` | Single stat |

**Screenshot of the Axiom dashboard:**

<p align="center">
  <img src="assets/Axiom results.jpeg" alt="Axiom Monitoring Dashboard" width="900"/>
</p>

> 📌 To recreate the dashboard: In Axiom, create a new dashboard, add a **Time series** chart per metric listed above using APL queries such as:
> ```
> ['nlp-project']
> | where ['_metric'] == 'sanad.rag.response_latency_ms'
> | summarize p95 = percentile(['_value'], 95) by bin_auto(_time)
> ```

### Live Dashboard (HyperDX)
The dashboard accurately visualizes all custom metrics, system health, and OpenTelemetry trace data in real-time.

<p align="center">
  <img src="assets/hyperdx.png" alt="HyperDX Live Dashboard" width="850"/>
</p>

---

## 🧪 Load Testing (Locust)

We utilize **Locust** for simulating concurrent user traffic to identify API bottlenecks.

**To run the load test:**
```bash
# Run with the web UI
uv run locust -f locustfile.py

# Or run headless (e.g., 50 users, spawn rate of 5/sec, for 1 minute)
uv run locust -f locustfile.py --headless -u 50 -r 5 --run-time 1m --host http://localhost:8000
```

### Load Testing Results
Below are the results of simulating concurrent user traffic against the API endpoints:

<p align="center">
  <img src="assets/locust-0.png" alt="Locust Test Run" width="850"/>
  <br>
  <img src="assets/locust-1.png" alt="Locust Request Statistics" width="850"/>
</p>

---

## 🔬 Model Comparison Table

We evaluated several Large Language Models on a set of standardized counseling queries. The comparison analyzes average inference latency, an aggregated clinical empathy score (1-5), and relevance.

| Model | Provider | Avg Latency | Empathy Score | Relevance Score | Cost (API) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **GPT-OSS-20B** | Groq | ~2.1s | 4.2 | 4.0 | Free Tier |
| **LLaMA-3-70B** | Groq | ~3.5s | 4.5 | 4.3 | Free Tier |
| **Gemma-2-9B** | Groq | ~1.2s | 3.8 | 3.5 | Free Tier |
| **Mixtral-8x7B**| Groq | ~2.8s | 4.1 | 4.1 | Free Tier |

> Note: We selected **GPT-OSS-20B** for our baseline as it provided the optimal balance of inference speed and emotional attunement.

---

## 🌐 Deployed API (AWS)

- **Backend API**: `https://sanad-ai.myvnc.com`
- **Swagger Docs**: `https://sanad-ai.myvnc.com/docs`
- **Frontend**: `https://mazen149.github.io/Mental-Health-RAG-Chatbot/`

---

## 👥 Team Members

This project was built and is maintained by:
1. **Ahmed Ashraf Abdulwahab Saleem**
2. **Mazen Mohamed Montaset Elsay**
3. **Peter Hany Fayez**

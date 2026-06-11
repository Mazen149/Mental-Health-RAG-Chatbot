import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """Centralized configuration and path management for Serene AI."""
    
    # ------------------------------------------------------------------------------
    # 1. Project Root & Environment Loading
    # ------------------------------------------------------------------------------
    _CURRENT_DIR = Path(__file__).resolve().parent
    
    # Dynamically find the project root by looking for .env or pyproject.toml
    PROJECT_ROOT = None
    for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
        if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
            PROJECT_ROOT = _parent
            break
    if PROJECT_ROOT is None:
        PROJECT_ROOT = _CURRENT_DIR.parent  # Fallback

    ENV_PATH = PROJECT_ROOT / ".env"
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        load_dotenv(override=True)


    # ------------------------------------------------------------------------------
    # 2. Derived File & Directory Paths (Cross-Platform)
    # ------------------------------------------------------------------------------
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    QDRANT_LOCAL_PATH = PROJECT_ROOT / "qdrant_db"
    CACHE_PATH = ARTIFACTS_DIR / "processed_docs.pkl"
    
    # Model 1 (Language Detection) Paths
    MOD1_VECTORIZER_PATH = ARTIFACTS_DIR / "langauge_detection" / "language_detection_best_vectorizer.pkl"
    MOD1_CLASSIFIER_PATH = ARTIFACTS_DIR / "langauge_detection" / "language_detection_best_model.pkl"
    
    # Model 2 (Emotion Classifier) Paths
    MOD2_DIR = ARTIFACTS_DIR / "emotion_classifier"
    MOD2_CONFIG_PATH = MOD2_DIR / "adapter_config.json"

    # ------------------------------------------------------------------------------
    # 3. Application Settings
    # ------------------------------------------------------------------------------
    ENABLE_TRANSLATION = os.getenv("ENABLE_TRANSLATION", "False").lower() in ("true", "1", "yes")
    LANGUAGE_DETECTION_MAX_LEN = 300

    # ------------------------------------------------------------------------------
    # 4. Model Names & Settings
    # ------------------------------------------------------------------------------
    GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-20b")
    GROQ_CLASSIFIER_MODEL = os.getenv("GROQ_CLASSIFIER_MODEL", "openai/gpt-oss-20b")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    EMOTION_BASE_MODEL = os.getenv("EMOTION_BASE_MODEL", "xlm-roberta-base")

    # ------------------------------------------------------------------------------
    # 5. API Keys & Connections
    # ------------------------------------------------------------------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "mental_health")

    # ------------------------------------------------------------------------------
    # 6. Demo Authentication
    # ------------------------------------------------------------------------------
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-secret-change-me")
    CHAT_DATABASE_PATH = ARTIFACTS_DIR / "chat_interactions.sqlite3"

# Expose a singleton instance
config = Config()

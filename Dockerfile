# =============================================================================
# SANAD AI — Multi-Stage Dockerfile
# =============================================================================
# Optimized for:
#   - Minimal image size (slim base, multi-stage build, no dev deps)
#   - Maximum layer caching (dependencies installed before source copy)
#   - Fast rebuilds (uv cache mount, bytecode compilation)
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder — install dependencies in an isolated environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ── Dependency installation (cached unless lock/toml change) ──────────────
# Copy ONLY dependency manifests first so this layer is cached across
# source-code-only changes.
COPY pyproject.toml uv.lock .python-version README.md ./

# Install production dependencies into a virtual-env at /app/.venv.
# --frozen ensures uv uses the lockfile exactly (no network resolution).
# --no-dev excludes dev-only packages (pytest, pre-commit, etc.).
# --mount=type=cache keeps uv's download cache across builds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ── Copy application source code ─────────────────────────────────────────
COPY src/ src/
COPY main.py ./

# Install the project itself (editable-like entry in the venv).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2: Runtime — lean production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Harden & optimise the runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Put the venv's bin on PATH so `python`, `uvicorn`, etc. resolve from it
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# ── Copy only what's needed from the builder ──────────────────────────────
COPY --from=builder /app/.venv .venv
COPY --from=builder /app/src   src
COPY --from=builder /app/main.py ./
COPY --from=builder /app/pyproject.toml ./

# Create directories that the app expects at runtime
# (artifacts/ will be populated by the HuggingFace downloader on first start)
RUN mkdir -p artifacts qdrant_db

# Non-root user for security — with a real home directory so that
# HuggingFace Hub, DSPy, and XET can write their runtime caches.
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

ENV HOME=/home/appuser
USER appuser

# Expose the API port
EXPOSE 8000

# Health-check against the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch the FastAPI server via uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

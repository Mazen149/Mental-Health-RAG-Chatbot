"""
================================================================================
SANAD AI — OpenTelemetry Metrics Module
================================================================================
Instruments the API with three metric categories exported to Axiom via OTLP:

  1. NLP/Model  : sanad.rag.response_latency_ms, sanad.rag.retrieval_score,
                  sanad.intent.count
  2. Data       : sanad.chat.message_length, sanad.feedback.votes
  3. Server     : sanad.server.request_count, sanad.server.error_count,
                  sanad.server.uptime_seconds, sanad.http.requests

Usage:
    from . import metrics
    metrics.init_metrics()          # call once at startup
    metrics.record_request()
    metrics.record_response_latency(ms)
    ...
================================================================================
"""

import os
import time

from opentelemetry import metrics
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level state — all None until init_metrics() is called
# ---------------------------------------------------------------------------
_start_time = time.time()
_meter = None

# NLP / Model
_response_latency_hist = None
_retrieval_score_hist = None
_intent_counter = None

# Data
_message_length_hist = None
_feedback_counter = None

# Server
_request_counter = None
_error_counter = None
_http_request_counter = None
_chat_request_counter = None


def init_metrics() -> None:
    """Initialize the OpenTelemetry MeterProvider and register all instruments.

    Reads configuration from environment variables:
      - OTEL_EXPORTER_OTLP_METRICS_ENDPOINT  (default: http://localhost:4318/v1/metrics)
      - AXIOM_API_TOKEN                        Bearer token for Axiom auth
      - AXIOM_METRICS_DATASET                  Axiom dataset name (default: nlp-project)
      - OTEL_METRIC_EXPORT_INTERVAL_MS         Export cadence in ms (default: 15000)

    When running with the OTel Collector (docker-compose), the endpoint points
    to the collector which then forwards to Axiom.  For direct-to-Axiom export
    (e.g. local dev), set the endpoint to https://api.axiom.co/v1/metrics.
    """
    global _meter
    global _response_latency_hist, _retrieval_score_hist, _intent_counter
    global _message_length_hist, _feedback_counter
    global _request_counter, _error_counter, _http_request_counter, _chat_request_counter

    # ------------------------------------------------------------------
    # Endpoint & auth headers
    # ------------------------------------------------------------------
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        "http://localhost:4318/v1/metrics",
    )

    headers: dict[str, str] = {}
    token = os.getenv("AXIOM_API_TOKEN", "")
    dataset = os.getenv("AXIOM_METRICS_DATASET", "nlp-project")

    # Only attach Axiom headers when sending directly (not via collector)
    # When going through the OTel Collector, the collector handles auth.
    if token and "axiom.co" in endpoint:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Axiom-Dataset"] = dataset

    # ------------------------------------------------------------------
    # Resource — identifies this service in Axiom
    # ------------------------------------------------------------------
    resource = Resource(attributes={
        SERVICE_NAME: "sanad-ai-backend",
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("ENVIRONMENT", "production"),
    })

    # ------------------------------------------------------------------
    # Exporter → Reader → Provider
    # ------------------------------------------------------------------
    exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers)
    export_interval_ms = int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000"))
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=export_interval_ms
    )
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter("sanad-ai", version="1.0.0")

    # ------------------------------------------------------------------
    # Metric 1 — NLP / Model
    # Rationale: Tracking RAG pipeline latency and retrieval scores lets
    # us detect model regressions, slow retrievals, or embedding quality
    # degradation over time.  Intent distribution reveals which user
    # needs dominate traffic (crisis vs. general questions).
    # ------------------------------------------------------------------
    _response_latency_hist = _meter.create_histogram(
        name="sanad.rag.response_latency_ms",
        unit="ms",
        description=(
            "End-to-end RAG pipeline latency in milliseconds — includes "
            "intent classification, hybrid retrieval, reranking, and LLM "
            "generation.  High p99 values indicate model or retrieval "
            "bottlenecks."
        ),
    )

    _retrieval_score_hist = _meter.create_histogram(
        name="sanad.rag.retrieval_score",
        description=(
            "Distribution of cosine-similarity reranker scores for the top "
            "retrieved counseling context chunks.  A shift toward lower "
            "scores signals embedding drift or dataset staleness."
        ),
    )

    _intent_counter = _meter.create_counter(
        name="sanad.intent.count",
        unit="1",
        description=(
            "Count of classified user intents broken down by label "
            "(asking_mental_health_question, crisis, greeting, out_of_scope, etc.). "
            "Monitors intent distribution and crisis-event frequency."
        ),
    )

    # ------------------------------------------------------------------
    # Metric 2 — Data
    # Rationale: Message length distribution catches anomalies such as
    # prompt injection (very long inputs) or bot abuse (unusually short
    # pings).  Feedback vote ratio (thumbs-up vs. down) is the primary
    # signal for response quality and RLHF data collection.
    # ------------------------------------------------------------------
    _message_length_hist = _meter.create_histogram(
        name="sanad.chat.message_length",
        unit="characters",
        description=(
            "Character-length distribution of user messages sent to /chat. "
            "Outliers flag potential prompt injection attempts or automated "
            "scraping activity."
        ),
    )

    _feedback_counter = _meter.create_counter(
        name="sanad.feedback.votes",
        unit="1",
        description=(
            "Count of user thumbs-up / thumbs-down votes on bot responses, "
            "labelled by {vote: up|down}.  The up/down ratio drives "
            "continuous RLHF improvement."
        ),
    )

    # ------------------------------------------------------------------
    # Metric 3 — Server
    # Rationale: Request count and error rate give basic service health
    # and SLA visibility.  Uptime tracks restarts.  Per-endpoint, per-
    # status-code HTTP counters enable drill-down on failures.
    # ------------------------------------------------------------------
    _request_counter = _meter.create_counter(
        name="sanad.server.request_count",
        unit="1",
        description=(
            "Total HTTP requests received by the API regardless of endpoint "
            "or outcome.  Used to compute overall throughput and load."
        ),
    )

    _error_counter = _meter.create_counter(
        name="sanad.server.error_count",
        unit="1",
        description=(
            "Total HTTP 5xx server-side errors.  A rising error rate "
            "compared to request count triggers on-call alerts."
        ),
    )

    _http_request_counter = _meter.create_counter(
        name="sanad.http.requests",
        unit="1",
        description=(
            "HTTP requests broken down by {method, endpoint, status_code} "
            "attributes.  Reveals which endpoints generate the most traffic "
            "or the highest error rates."
        ),
    )

    _chat_request_counter = _meter.create_counter(
        name="sanad.chat.requests",
        unit="1",
        description=(
            "Total calls to /chat and /chat/stream — the primary business "
            "activity counters for the chatbot."
        ),
    )

    def _uptime_cb(_options):
        yield Observation(time.time() - _start_time)

    try:
        _meter.create_observable_gauge(
            name="sanad.server.uptime_seconds",
            callbacks=[_uptime_cb],
            description=(
                "Server uptime in seconds since the last startup.  A sudden "
                "drop to near zero indicates an unexpected restart."
            ),
        )
    except Exception:
        # Never let uptime registration crash the server.
        pass


# ---------------------------------------------------------------------------
# Public wrapper functions
# All functions silently no-op when metrics have not been initialized.
# ---------------------------------------------------------------------------

def record_request(count: int = 1) -> None:
    """Increment the total request counter."""
    if _request_counter is None:
        return
    try:
        _request_counter.add(count)
    except Exception:
        pass


def record_error(count: int = 1) -> None:
    """Increment the server error counter."""
    if _error_counter is None:
        return
    try:
        _error_counter.add(count)
    except Exception:
        pass


def record_http_request(method: str, endpoint: str, status_code: str) -> None:
    """Record an HTTP request with method / endpoint / status_code labels."""
    if _http_request_counter is None:
        return
    try:
        _http_request_counter.add(
            1,
            {"method": method, "endpoint": endpoint, "status_code": status_code},
        )
    except Exception:
        pass


def record_response_latency(ms: float) -> None:
    """Record end-to-end RAG pipeline latency in milliseconds."""
    if _response_latency_hist is None:
        return
    try:
        _response_latency_hist.record(ms)
    except Exception:
        pass


def record_rerank_score(score: float) -> None:
    """Record a cosine-similarity retrieval / reranker score."""
    if _retrieval_score_hist is None:
        return
    try:
        _retrieval_score_hist.record(score)
    except Exception:
        pass


def record_intent(intent: str) -> None:
    """Increment the intent distribution counter for *intent*."""
    if _intent_counter is None:
        return
    try:
        _intent_counter.add(1, {"intent": intent})
    except Exception:
        pass


def record_message_length(length: int) -> None:
    """Record the character length of a user message."""
    if _message_length_hist is None:
        return
    try:
        _message_length_hist.record(length)
    except Exception:
        pass


def record_feedback_vote(vote: str) -> None:
    """Increment the feedback vote counter.  *vote* should be 'up' or 'down'."""
    if _feedback_counter is None:
        return
    try:
        _feedback_counter.add(1, {"vote": vote})
    except Exception:
        pass


def record_chat_request(count: int = 1) -> None:
    """Increment the /chat endpoint request counter."""
    if _chat_request_counter is None:
        return
    try:
        _chat_request_counter.add(count)
    except Exception:
        pass
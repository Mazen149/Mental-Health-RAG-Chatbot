import time
import os
from urllib.parse import urlparse, urlunparse

from opentelemetry import metrics
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from dotenv import load_dotenv

load_dotenv()

from .config import config

_start_time = time.time()
_meter = None
_request_counter = None
_error_counter = None
_response_latency_hist = None
_message_length_hist = None
_retrieval_score_hist = None
_feedback_counter = None


def init_metrics() -> None:
    """Initialize OpenTelemetry metrics exporter (OTLP HTTP).

    Required env vars:
      AXIOM_API_TOKEN or AXIOM_ACCESS_TOKEN  — Axiom API token
      AXIOM_METRICS_DATASET                  — Axiom metrics dataset name
    Optional:
      OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or AXIOM_OTLP_METRICS_ENDPOINT
          — defaults to https://api.axiom.co/v1/metrics
    """
    global _meter, _request_counter, _error_counter, _response_latency_hist
    global _message_length_hist, _retrieval_score_hist, _feedback_counter

    # --- Endpoint resolution ---
    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        os.getenv("AXIOM_OTLP_METRICS_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "https://api.axiom.co",
    )

    # Normalize: ensure path ends with /v1/metrics
    parsed = urlparse(endpoint)
    path = (parsed.path or "").rstrip("/")
    if path not in ("/v1/metrics", "/otlp/v1/metrics"):
        path = path + "/v1/metrics"
    endpoint = urlunparse(parsed._replace(path=path))

    # --- Headers ---
    headers = {}

    token = os.getenv("AXIOM_API_TOKEN") or os.getenv("AXIOM_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dataset = os.getenv("AXIOM_METRICS_DATASET")
    if dataset:
        headers["X-Axiom-Metrics-Dataset"] = dataset

    # --- Provider setup ---
    exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers)

    export_interval = int(
        os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "5000")
    )
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=export_interval,
    )
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _meter = metrics.get_meter(__name__)

    # --- Instrument definitions ---

    # Server metrics
    _request_counter = _meter.create_counter(
        "server.request.count",
        description="Total HTTP requests",
    )
    _error_counter = _meter.create_counter(
        "server.error.count",
        description="Total HTTP errors",
    )
    _response_latency_hist = _meter.create_histogram(
        "server.response.latency_ms",
        unit="ms",
        description="HTTP response latency (ms)",
    )

    # Data & model metrics
    _message_length_hist = _meter.create_histogram(
        "chat.message.length",
        description="Distribution of user message lengths (tokens/chars)",
    )
    _retrieval_score_hist = _meter.create_histogram(
        "rag.retrieval.score",
        description="Distribution of retrieval/reranker scores",
    )

    # Feedback votes counter (labelled by vote: up/down)
    _feedback_counter = _meter.create_counter(
        "feedback.votes",
        description="Count of feedback votes (labelled by vote)",
    )

    # FIX 3: Simplified uptime callback — removed the dead old-style observer
    # shim. The OTel Python SDK 1.x+ only uses the new-style API: callbacks
    # must return an iterable of Observation objects.
    def _uptime_cb(_options):
        yield Observation(time.time() - _start_time)

    try:
        _meter.create_observable_gauge(
            "server.uptime_seconds",
            callbacks=[_uptime_cb],
            description="Server uptime in seconds",
        )
    except Exception:
        # Failing to register uptime should never break app startup.
        pass


# ---------------------------------------------------------------------------
# Public wrappers — guard against missing initialization
# ---------------------------------------------------------------------------

def record_request(count: int = 1) -> None:
    if _request_counter is None:
        return
    try:
        _request_counter.add(count)
    except Exception:
        pass


def record_error(count: int = 1) -> None:
    if _error_counter is None:
        return
    try:
        _error_counter.add(count)
    except Exception:
        pass


def record_response_latency(ms: float) -> None:
    if _response_latency_hist is None:
        return
    try:
        _response_latency_hist.record(ms)
    except Exception:
        pass


def record_message_length(length: int) -> None:
    if _message_length_hist is None:
        return
    try:
        _message_length_hist.record(length)
    except Exception:
        pass


def record_rerank_score(score: float) -> None:
    if _retrieval_score_hist is None:
        return
    try:
        _retrieval_score_hist.record(score)
    except Exception:
        pass


def record_feedback_vote(vote: str) -> None:
    if _feedback_counter is None:
        return
    try:
        _feedback_counter.add(1, {"vote": vote})
    except Exception:
        pass
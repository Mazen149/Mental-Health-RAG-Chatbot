import time
import os
from typing import Optional

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

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
    """Initialize OpenTelemetry metrics exporter (OTLP HTTP). Configure endpoint via
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT or AXIOM_OTLP_METRICS_ENDPOINT and token via
    AXIOM_API_TOKEN / AXIOM_ACCESS_TOKEN environment variables."""
    global _meter, _request_counter, _error_counter, _response_latency_hist
    global _message_length_hist, _retrieval_score_hist, _feedback_counter

    endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
        os.getenv("AXIOM_OTLP_METRICS_ENDPOINT", "http://localhost:4318/v1/metrics"),
    )
    headers = {}
    token = os.getenv("AXIOM_API_TOKEN") or os.getenv("AXIOM_ACCESS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    exporter = OTLPMetricExporter(endpoint=endpoint, headers=headers)
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _meter = metrics.get_meter(__name__)

    # Server metrics
    _request_counter = _meter.create_counter(
        "server.request.count", description="Total HTTP requests"
    )
    _error_counter = _meter.create_counter(
        "server.error.count", description="Total HTTP errors"
    )
    _response_latency_hist = _meter.create_histogram(
        "server.response.latency_ms", unit="ms", description="HTTP response latency (ms)"
    )

    # Data & model metrics
    _message_length_hist = _meter.create_histogram(
        "chat.message.length", description="Distribution of user message lengths (tokens/chars)"
    )
    _retrieval_score_hist = _meter.create_histogram(
        "rag.retrieval.score", description="Distribution of retrieval/reranker scores"
    )

    # Feedback votes counter (labelled by vote: up/down)
    _feedback_counter = _meter.create_counter(
        "feedback.votes", description="Count of feedback votes (labelled by vote)"
    )

    # Uptime observable gauge
    def _uptime_cb(observer):
        try:
            observer.observe(time.time() - _start_time, {})
        except Exception:
            pass

    try:
        _meter.create_observable_gauge(
            "server.uptime_seconds",
            callbacks=[_uptime_cb],
            description="Server uptime in seconds",
        )
    except Exception:
        # Some SDK versions may differ; failing to register uptime shouldn't break app.
        pass


# Small wrappers to guard against missing initialization
def record_request(count: int = 1) -> None:
    if _request_counter is None:
        return
    try:
        _request_counter.add(count, {})
    except Exception:
        pass


def record_error(count: int = 1) -> None:
    if _error_counter is None:
        return
    try:
        _error_counter.add(count, {})
    except Exception:
        pass


def record_response_latency(ms: float) -> None:
    if _response_latency_hist is None:
        return
    try:
        _response_latency_hist.record(ms, {})
    except Exception:
        pass


def record_message_length(length: int) -> None:
    if _message_length_hist is None:
        return
    try:
        _message_length_hist.record(length, {})
    except Exception:
        pass


def record_rerank_score(score: float) -> None:
    if _retrieval_score_hist is None:
        return
    try:
        _retrieval_score_hist.record(score, {})
    except Exception:
        pass


def record_feedback_vote(vote: str) -> None:
    if _feedback_counter is None:
        return
    try:
        # Use attribute label to distinguish up/down votes
        _feedback_counter.add(1, {"vote": vote})
    except Exception:
        pass

import time
import os
from typing import Optional
from urllib.parse import urlparse, urlunparse

from opentelemetry import metrics
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

    # Normalize common OTLP endpoints to ensure the metrics collector path is
    # included. This prevents 404 errors from users passing base OTLP host URLs.
    parsed = urlparse(endpoint)
    path = parsed.path or ""
    if path.rstrip("/") not in ("/v1/metrics", "/otlp/v1/metrics"):
        normalized_path = path.rstrip("/") + "/v1/metrics"
        endpoint = urlunparse(
            parsed._replace(path=normalized_path)
        )

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
    def _uptime_cb(callback_arg):
        """Compatibility shim for OpenTelemetry observable callbacks.

        Older SDKs pass an observer object with an `observe(value, attributes)`
        method. Newer SDKs expect the callback to return an iterable of objects
        with a `.value` attribute. This function handles both cases.
        """
        try:
            value = time.time() - _start_time

            # Old-style observer API: object with `observe` method
            observe = getattr(callback_arg, "observe", None)
            if callable(observe):
                try:
                    callback_arg.observe(value, {})
                except Exception:
                    # Swallow to avoid breaking app startup
                    pass
                # Old API does not expect a return value
                return

            # New-style API: return an iterable of lightweight objects
            from types import SimpleNamespace

            # Provide `attributes` and `context` for compatibility with
            # OpenTelemetry SDK expectations (api_measurement.attributes)
            return (SimpleNamespace(value=value, attributes={}, context=None),)
        except Exception:
            # Ensure callback never raises; return empty iterable as fallback
            return ()

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

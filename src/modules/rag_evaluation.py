from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from langsmith import Client
from langsmith import traceable

try:
    from .rag import MentalHealthRAG
except ImportError:
    _CURRENT_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _CURRENT_FILE.parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.modules.rag import MentalHealthRAG


# -----------------------------------------------------------------------------
# Environment bootstrap
# -----------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = None
for _parent in [_CURRENT_DIR] + list(_CURRENT_DIR.parents):
    if (_parent / ".env").exists() or (_parent / "pyproject.toml").exists():
        _PROJECT_ROOT = _parent
        break
if _PROJECT_ROOT is None:
    _PROJECT_ROOT = _CURRENT_DIR.parent

_ENV_PATH = _PROJECT_ROOT / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()


# -----------------------------------------------------------------------------
# LangSmith + judge model setup
# -----------------------------------------------------------------------------
DEFAULT_JUDGE_MODEL = os.getenv(
    "GROQ_EVALUATION_MODEL", os.getenv("EVALUATION_LLM_MODEL", "llama-3.1-8b-instant")
)
DEFAULT_EVAL_PROJECT = os.getenv("LANGSMITH_PROJECT", "serene-rag-evaluation")
DEFAULT_EVAL_EXPERIMENT_PREFIX = os.getenv(
    "LANGSMITH_EXPERIMENT_PREFIX", "serene-rag-eval"
)
DEFAULT_DATASET_UPLOAD_BATCH_SIZE = int(os.getenv("LANGSMITH_EXAMPLE_BATCH_SIZE", "25"))
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("EVALUATION_GROQ_API_KEY")
        or os.getenv("LANGSMITH_EVALUATION_GROQ_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "A Groq API key is required for evaluation. Set GROQ_API_KEY or EVALUATION_GROQ_API_KEY."
        )

    _groq_client = Groq(api_key=api_key)
    return _groq_client


@dataclass
class MetricResult:
    score: float
    reason: str
    verdict: str | None = None


def _coerce_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def _format_contexts(contexts: Iterable[str], limit: int = 5) -> str:
    chunks = []
    for index, context in enumerate(list(contexts)[:limit], start=1):
        chunks.append(f"Context [{index}]: {context}")
    return "\n\n".join(chunks)


@traceable(name="rag.eval.judge_call", run_type="llm")
def _call_judge(prompt: str, *, model: str = DEFAULT_JUDGE_MODEL) -> MetricResult:
    response = _get_groq_client().chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict RAG evaluation judge. "
                    "Return valid JSON only with keys: score, verdict, reason. "
                    "score must be a number between 0 and 1."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_text = response.choices[0].message.content or "{}"
    payload = _extract_json(raw_text)
    score = _coerce_score(payload.get("score", 0.0))
    verdict = payload.get("verdict")
    reason = str(payload.get("reason", "")).strip() or raw_text.strip()
    return MetricResult(score=score, reason=reason, verdict=verdict)


@traceable(name="rag.eval.retrieval_relevance", run_type="llm")
def score_retrieval_relevance(
    question: str,
    retrieved_contexts: list[str],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
) -> MetricResult:
    prompt = f"""
Question:
{question}

Retrieved documents:
{_format_contexts(retrieved_contexts)}

Judge how relevant the retrieved documents are to answering the question.
Give a score from 0 to 1 where:
0 = irrelevant, 0.5 = partially relevant, 1 = highly relevant.
Return JSON with keys score, verdict, reason.
"""
    return _call_judge(prompt, model=model)


@traceable(name="rag.eval.groundedness", run_type="llm")
def score_groundedness(
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    *,
    model: str = DEFAULT_JUDGE_MODEL,
) -> MetricResult:
    prompt = f"""
Question:
{question}

Answer:
{answer}

Retrieved documents:
{_format_contexts(retrieved_contexts)}

Judge whether the answer is grounded in the retrieved documents.
Give a score from 0 to 1 where:
0 = unsupported or hallucinated, 0.5 = partially supported, 1 = fully supported by the documents.
Return JSON with keys score, verdict, reason.
"""
    return _call_judge(prompt, model=model)


@traceable(name="rag.eval.answer_relevance", run_type="llm")
def score_answer_relevance(
    question: str,
    answer: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
) -> MetricResult:
    prompt = f"""
Question:
{question}

Answer:
{answer}

Judge whether the answer directly addresses the question and stays on topic.
Give a score from 0 to 1 where:
0 = off-topic, 0.5 = somewhat relevant, 1 = directly answers the question.
Return JSON with keys score, verdict, reason.
"""
    return _call_judge(prompt, model=model)


@traceable(name="rag.eval.correctness", run_type="llm")
def score_correctness(
    question: str,
    answer: str,
    reference_answer: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
) -> MetricResult:
    prompt = f"""
Question:
{question}

Reference answer:
{reference_answer}

Predicted answer:
{answer}

Judge whether the predicted answer matches the reference answer in meaning and clinical guidance.
Give a score from 0 to 1 where:
0 = incorrect, 0.5 = partially correct, 1 = correct and consistent with the reference answer.
Return JSON with keys score, verdict, reason.
"""
    return _call_judge(prompt, model=model)


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    question = sample.get("question") or sample.get("query") or sample.get("input")
    if not question:
        raise ValueError("Each sample must include a question/query/input field.")

    reference_answer = (
        sample.get("reference_answer")
        or sample.get("ground_truth")
        or sample.get("expected_answer")
        or sample.get("answer")
    )
    if reference_answer is None:
        raise ValueError(
            "Each sample must include a reference_answer, ground_truth, expected_answer, or answer field."
        )

    history = sample.get("history") or []
    if isinstance(history, str):
        history = json.loads(history)

    return {
        "question": str(question).strip(),
        "reference_answer": str(reference_answer).strip(),
        "history": history,
    }


def _build_langsmith_examples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for sample in samples:
        normalized = _normalize_sample(sample)
        examples.append(
            {
                "inputs": {
                    "question": normalized["question"],
                    "history": normalized["history"] or [],
                },
                "outputs": {
                    "reference_answer": normalized["reference_answer"],
                },
                "metadata": {
                    "source": "local_eval_file",
                },
            }
        )
    return examples


def _chunked(
    items: list[dict[str, Any]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _ensure_langsmith_dataset(
    client: Client,
    *,
    dataset_name: str,
    samples: list[dict[str, Any]],
    description: str | None = None,
    upload_batch_size: int = DEFAULT_DATASET_UPLOAD_BATCH_SIZE,
) -> Any:
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description or "Serene AI RAG evaluation dataset.",
            metadata={"source": "src/modules/rag_evaluation.py"},
        )

    try:
        existing_examples = list(client.list_examples(dataset_name=dataset_name))
    except Exception:
        existing_examples = []

    existing_count = len(existing_examples)
    examples = _build_langsmith_examples(samples)
    if existing_count >= len(examples):
        print(
            f"--> [LangSmith] Dataset '{dataset_name}' already has {existing_count} examples; "
            "skipping upload."
        )
        return dataset

    if existing_count > 0:
        print(
            f"--> [LangSmith] Dataset '{dataset_name}' already has {existing_count} examples; "
            f"resuming upload from example {existing_count + 1}."
        )
        examples = examples[existing_count:]

    total_batches = max(1, (len(examples) + upload_batch_size - 1) // upload_batch_size)
    for batch_number, batch in enumerate(
        _chunked(examples, upload_batch_size), start=1
    ):
        print(
            f"--> [LangSmith] Uploading dataset examples batch {batch_number}/{total_batches} "
            f"({len(batch)} examples)..."
        )
        client.create_examples(
            dataset_name=dataset_name,
            examples=batch,
        )

    return dataset


def load_eval_samples(source: str | Path) -> list[dict[str, Any]]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
        return [_normalize_sample(record) for record in df.to_dict(orient="records")]

    if path.suffix.lower() in {".jsonl", ".json"}:
        records: list[dict[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(_normalize_sample(json.loads(line)))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("samples", [])
            if not isinstance(payload, list):
                raise ValueError(
                    "JSON dataset must be a list of samples or an object with a 'samples' list."
                )
            records = [_normalize_sample(item) for item in payload]
        return records

    raise ValueError("Unsupported dataset format. Use CSV, JSON, or JSONL.")


def _build_rag_for_eval() -> MentalHealthRAG:
    rag = MentalHealthRAG()

    @traceable(name="rag.eval.build_index", run_type="chain")
    def _initialize() -> int:
        documents = rag.load_and_preprocess()
        rag.setup_retriever(documents)
        return len(documents)

    document_count = _initialize()
    print(f"--> [RAG Eval] Loaded {document_count} documents for evaluation.")
    return rag


def _format_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]] | None:
    if not history:
        return None

    normalized_history: list[dict[str, str]] = []
    for message in history:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            normalized_history.append({"role": role, "content": str(content)})
    return normalized_history or None


def _make_target(rag: MentalHealthRAG):
    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", "")).strip()
        if not question:
            raise ValueError("Question is required.")

        history = _format_history(inputs.get("history"))
        result = rag.query(
            user_query=question,
            translated_query=question,
            history=history,
        )
        resources = result.get("resources", []) or []

        # ✅ FIXED: use "response" — this is what was actually injected into the LLM prompt
        retrieved_contexts = [
            str(resource.get("response", ""))
            for resource in resources
            if resource.get("response")
        ]

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "resources": resources,
            "retrieved_contexts": retrieved_contexts,
            "intent": result.get("intent"),
            "language": result.get("language"),
            "emotion": result.get("emotion"),
        }

    return target


def retrieval_relevance_evaluator(run, example) -> dict[str, Any]:
    inputs = run.inputs or {}
    outputs = run.outputs or {}
    question = str(inputs.get("question", "")).strip()
    retrieved_contexts = outputs.get("retrieved_contexts", []) or []
    metric = score_retrieval_relevance(question, retrieved_contexts)
    return {
        "key": "retrieval_relevance",
        "score": metric.score,
        "comment": metric.reason,
    }


def groundedness_evaluator(run, example) -> dict[str, Any]:
    inputs = run.inputs or {}
    outputs = run.outputs or {}
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    retrieved_contexts = outputs.get("retrieved_contexts", []) or []
    metric = score_groundedness(question, answer, retrieved_contexts)
    return {"key": "groundedness", "score": metric.score, "comment": metric.reason}


def answer_relevance_evaluator(run, example) -> dict[str, Any]:
    inputs = run.inputs or {}
    outputs = run.outputs or {}
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    metric = score_answer_relevance(question, answer)
    return {"key": "answer_relevance", "score": metric.score, "comment": metric.reason}


def correctness_evaluator(run, example) -> dict[str, Any]:
    inputs = run.inputs or {}
    outputs = run.outputs or {}
    question = str(inputs.get("question", "")).strip()
    answer = str(outputs.get("answer", "")).strip()
    reference_answer = str((example.outputs or {}).get("reference_answer", "")).strip()
    metric = score_correctness(question, answer, reference_answer)
    return {"key": "correctness", "score": metric.score, "comment": metric.reason}


def run_evaluation(
    dataset_path: str | Path,
    *,
    output_path: str | Path | None = None,
    limit: int | None = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    project_name: str = DEFAULT_EVAL_PROJECT,
    experiment_prefix: str = DEFAULT_EVAL_EXPERIMENT_PREFIX,
    upload_batch_size: int = DEFAULT_DATASET_UPLOAD_BATCH_SIZE,
) -> pd.DataFrame:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project_name)

    samples = load_eval_samples(dataset_path)
    if limit is not None:
        samples = samples[:limit]
        print(f"--> [RAG Eval] Limiting evaluation to {len(samples)} samples.")
    rag = _build_rag_for_eval()
    client = Client()
    dataset_name = f"{experiment_prefix}-{Path(dataset_path).stem}"
    _ensure_langsmith_dataset(
        client,
        dataset_name=dataset_name,
        samples=samples,
        description=f"Evaluation dataset generated from {Path(dataset_path).name}.",
        upload_batch_size=upload_batch_size,
    )

    target = _make_target(rag)
    evaluators = [
        retrieval_relevance_evaluator,
        groundedness_evaluator,
        answer_relevance_evaluator,
        correctness_evaluator,
    ]
    try:
        experiment_results = client.evaluate(
            target,
            data=dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            description=(
                "Serene AI RAG evaluation with retrieval relevance, groundedness, "
                "answer relevance, and correctness."
            ),
            metadata={
                "dataset_path": str(Path(dataset_path).resolve()),
                "judge_model": judge_model,
            },
            blocking=True,
        )
    finally:
        rag.close()

    rows: list[dict[str, Any]] = []
    for item in experiment_results:
        evaluation_scores = {}
        evaluation_comments = {}
        for result in item["evaluation_results"]["results"]:
            evaluation_scores[result.key] = result.score
            evaluation_comments[result.key] = result.comment

        run_outputs = item["run"].outputs or {}
        rows.append(
            {
                "question": item["example"].inputs.get("question"),
                "reference_answer": (item["example"].outputs or {}).get(
                    "reference_answer"
                ),
                "predicted_answer": run_outputs.get("answer"),
                "retrieved_contexts": run_outputs.get("retrieved_contexts", []),
                "intent": run_outputs.get("intent"),
                "language": run_outputs.get("language"),
                "emotion": run_outputs.get("emotion"),
                "retrieval_relevance": evaluation_scores.get(
                    "retrieval_relevance", 0.0
                ),
                "retrieval_relevance_reason": evaluation_comments.get(
                    "retrieval_relevance", ""
                ),
                "groundedness": evaluation_scores.get("groundedness", 0.0),
                "groundedness_reason": evaluation_comments.get("groundedness", ""),
                "answer_relevance": evaluation_scores.get("answer_relevance", 0.0),
                "answer_relevance_reason": evaluation_comments.get(
                    "answer_relevance", ""
                ),
                "correctness": evaluation_scores.get("correctness", 0.0),
                "correctness_reason": evaluation_comments.get("correctness", ""),
            }
        )

    results = pd.DataFrame(rows)

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            results.to_csv(out_path, index=False)
        else:
            out_path.write_text(
                results.to_json(orient="records", indent=2, force_ascii=False),
                encoding="utf-8",
            )

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the Serene AI RAG system with LangSmith tracing."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a CSV, JSON, or JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for the evaluation results (.csv or .json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model to use for scoring metrics (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_EVAL_PROJECT,
        help=f"LangSmith project name (default: {DEFAULT_EVAL_PROJECT}).",
    )
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=DEFAULT_DATASET_UPLOAD_BATCH_SIZE,
        help=f"LangSmith dataset example upload batch size (default: {DEFAULT_DATASET_UPLOAD_BATCH_SIZE}).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    results = run_evaluation(
        args.dataset,
        output_path=args.output,
        limit=args.limit,
        judge_model=args.judge_model,
        project_name=args.project,
        upload_batch_size=args.upload_batch_size,
    )

    summary = results[
        [
            "retrieval_relevance",
            "groundedness",
            "answer_relevance",
            "correctness",
        ]
    ].mean(numeric_only=True)

    print("\nEvaluation summary:")
    for metric, value in summary.items():
        print(f"- {metric}: {value:.3f}")

    if args.output:
        print(f"\nSaved detailed results to: {args.output}")


if __name__ == "__main__":
    main()

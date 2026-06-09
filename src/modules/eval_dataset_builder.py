from __future__ import annotations

import argparse
import json
import pickle
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "artifacts" / "processed_docs.pkl"


@dataclass
class EvalSample:
    question: str
    reference_answer: str
    history: list[dict[str, Any]]
    source: str
    source_id: str | None = None
    intent: str | None = None
    language: str | None = None
    needs_annotation: bool = False


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _load_cached_documents(cache_path: Path = DEFAULT_CACHE_PATH) -> list[Any]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Processed document cache not found: {cache_path}. "
            "Run the RAG preprocessing step first."
        )

    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict):
        documents = payload.get("documents", [])
    else:
        documents = payload

    if not isinstance(documents, list):
        raise ValueError("Processed document cache is malformed.")

    return documents


def build_eval_samples_from_documents(
    *,
    cache_path: Path = DEFAULT_CACHE_PATH,
    limit: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[EvalSample]:
    documents = _load_cached_documents(cache_path)

    rows: list[EvalSample] = []
    seen: set[tuple[str, str]] = set()

    for index, document in enumerate(documents):
        metadata = getattr(document, "metadata", {}) or {}
        question = _normalize_text(metadata.get("question"))
        reference_answer = _normalize_text(metadata.get("response"))

        if not question or not reference_answer:
            page_content = _normalize_text(getattr(document, "page_content", ""))
            if "\n\n" in page_content:
                inferred_question, inferred_answer = page_content.split("\n\n", 1)
                question = question or _normalize_text(inferred_question)
                reference_answer = reference_answer or _normalize_text(inferred_answer)

        if not question or not reference_answer:
            continue

        key = (question.lower(), reference_answer.lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            EvalSample(
                question=question,
                reference_answer=reference_answer,
                history=[],
                source="processed_docs_cache",
                source_id=str(index),
            )
        )

    if shuffle:
        random.Random(seed).shuffle(rows)

    if limit is not None:
        rows = rows[:limit]

    return rows


def build_review_samples_from_interactions(
    *,
    database_path: Path = DATABASE_PATH,
    limit: int | None = None,
    only_annotated: bool = False,
) -> list[EvalSample]:
    if not database_path.exists():
        raise FileNotFoundError(f"Interaction database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        query = """
            SELECT id, query, answer, language, intent, history_json, created_at
            FROM interactions
            ORDER BY id ASC
        """
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()

    samples: list[EvalSample] = []
    for row in rows:
        question = _normalize_text(row["query"])
        answer = _normalize_text(row["answer"])
        if not question:
            continue

        history: list[dict[str, Any]] = []
        if row["history_json"]:
            try:
                parsed_history = json.loads(row["history_json"])
                if isinstance(parsed_history, list):
                    history = parsed_history
            except json.JSONDecodeError:
                history = []

        sample = EvalSample(
            question=question,
            reference_answer=answer,
            history=history,
            source="interaction_log",
            source_id=str(row["id"]),
            intent=_normalize_text(row["intent"]) or None,
            language=_normalize_text(row["language"]) or None,
            needs_annotation=True,
        )

        if only_annotated and not sample.reference_answer:
            continue

        samples.append(sample)

    if limit is not None:
        samples = samples[:limit]

    return samples


def write_jsonl(samples: list[EvalSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = asdict(sample)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(samples: list[EvalSample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(sample) for sample in samples]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Serene AI evaluation dataset.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the dataset (.jsonl or .json).",
    )
    parser.add_argument(
        "--source",
        choices=["documents", "interactions"],
        default="documents",
        help="Where to build the dataset from.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of samples to export.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep document samples in their original order.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_path = Path(args.output)
    if args.source == "documents":
        samples = build_eval_samples_from_documents(
            limit=args.limit,
            shuffle=not args.no_shuffle,
        )
    else:
        samples = build_review_samples_from_interactions(limit=args.limit)

    if output_path.suffix.lower() == ".json":
        write_json(samples, output_path)
    else:
        write_jsonl(samples, output_path)

    df = pd.DataFrame([asdict(sample) for sample in samples])
    print(f"Exported {len(samples)} samples to {output_path}")
    if not df.empty:
        print(df[["question", "source"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()

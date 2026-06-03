"""
Combined RAGAS and DeepEval evaluation runner for Mental Health RAG.
Generates comparison report and visualizations.
"""

import os
import sys
import json
import argparse
from typing import List, Optional, Dict
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Handle both direct execution and module import
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_pipeline import MentalHealthRAG
    from evaluator_ragas import RAGASEvaluator
    from evaluator_deepeval import DeepEvalEvaluator
else:
    from .rag_pipeline import MentalHealthRAG
    from .evaluator_ragas import RAGASEvaluator
    from .evaluator_deepeval import DeepEvalEvaluator


def load_test_dataset(filepath: Optional[str] = None) -> List[str]:
    """Load test queries from file or use defaults."""
    default_queries = [
        "i am depressed, what should i do?",
        "How can I manage anxiety?",
        "What are signs of depression?",
        "How do I handle stress at work?",
        "Can you explain burnout?",
        "How to improve sleep quality?",
        "What is mindfulness?",
        "How to deal with grief?",
        "انا أشعر بالقلق، ماذا أفعل؟",
        "كيف أتعامل مع الإجهاد؟",
    ]

    if filepath and os.path.exists(filepath):
        with open(filepath, "r") as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries or default_queries
    return default_queries


def run_ragas_evaluation(
    rag: MentalHealthRAG, test_queries: List[str], output_dir: str
) -> Dict:
    """Run RAGAS evaluation."""
    print("\n" + "=" * 80)
    print("STARTING RAGAS EVALUATION")
    print("=" * 80)

    try:
        evaluator = RAGASEvaluator(rag)
        results = evaluator.evaluate(
            test_queries=test_queries,
            output_file=os.path.join(output_dir, "ragas_results.json"),
        )
        evaluator.print_summary(results)

        return {
            "status": "success",
            "metrics": {
                "context_precision": results.context_precision,
                "context_recall": results.context_recall,
                "faithfulness": results.faithfulness,
                "answer_relevancy": results.answer_relevancy,
            },
        }
    except Exception as e:
        print(f"ERROR in RAGAS evaluation: {e}")
        return {"status": "failed", "error": str(e)}


def run_deepeval_evaluation(
    rag: MentalHealthRAG, test_queries: List[str], output_dir: str
) -> Dict:
    """Run DeepEval evaluation."""
    print("\n" + "=" * 80)
    print("STARTING DEEPEVAL EVALUATION")
    print("=" * 80)

    try:
        evaluator = DeepEvalEvaluator(rag)
        results = evaluator.evaluate(
            test_queries=test_queries,
            output_file=os.path.join(output_dir, "deepeval_results.json"),
        )
        evaluator.print_summary(results)

        return {
            "status": "success",
            "metrics": {
                "faithfulness": results.faithfulness,
                "answer_relevancy": results.answer_relevancy,
                "contextual_relevancy": results.contextual_relevancy,
                "contextual_precision": results.contextual_precision,
            },
        }
    except Exception as e:
        print(f"ERROR in DeepEval evaluation: {e}")
        return {"status": "failed", "error": str(e)}


def generate_comparison_report(
    ragas_results: Dict, deepeval_results: Dict, output_file: str
) -> None:
    """Generate comparison report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "ragas": ragas_results,
        "deepeval": deepeval_results,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nComparison report saved: {output_file}")


def plot_metrics(
    ragas_results: Dict, deepeval_results: Dict, output_dir: str
) -> None:
    """Generate comparison plots."""
    if ragas_results["status"] == "failed" or deepeval_results["status"] == "failed":
        print("Skipping plot generation due to evaluation failures.")
        return

    ragas_metrics = ragas_results.get("metrics", {})
    deepeval_metrics = deepeval_results.get("metrics", {})

    # Normalize metrics to 0-1 scale for comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RAG Evaluation Metrics: RAGAS vs DeepEval", fontsize=16)

    # Common metrics
    common_metrics = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
    ]

    for idx, (metric_key, metric_label) in enumerate(common_metrics):
        ax = axes[idx // 2, idx % 2]

        ragas_val = ragas_metrics.get(metric_key, 0)
        deepeval_val = deepeval_metrics.get(metric_key, 0)

        x = np.arange(2)
        width = 0.35

        bars1 = ax.bar(x - width / 2, [ragas_val], width, label="RAGAS", alpha=0.8)
        bars2 = ax.bar(x + width / 2, [deepeval_val], width, label="DeepEval", alpha=0.8)

        ax.set_ylabel("Score")
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(["Score"])
        ax.legend()
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    # RAGAS unique metrics
    ax = axes[0, 1]
    ragas_unique = [
        ragas_metrics.get("context_precision", 0),
        ragas_metrics.get("context_recall", 0),
    ]
    ax.barh(["Context Precision", "Context Recall"], ragas_unique, alpha=0.8, color="skyblue")
    ax.set_xlabel("Score")
    ax.set_title("RAGAS Specific Metrics")
    ax.set_xlim(0, 1)
    for i, v in enumerate(ragas_unique):
        ax.text(v, i, f" {v:.3f}", va="center")

    # DeepEval unique metrics
    ax = axes[1, 1]
    deepeval_unique = [
        deepeval_metrics.get("contextual_relevancy", 0),
        deepeval_metrics.get("contextual_precision", 0),
    ]
    ax.barh(["Contextual Relevancy", "Contextual Precision"], deepeval_unique, alpha=0.8, color="lightcoral")
    ax.set_xlabel("Score")
    ax.set_title("DeepEval Specific Metrics")
    ax.set_xlim(0, 1)
    for i, v in enumerate(deepeval_unique):
        ax.text(v, i, f" {v:.3f}", va="center")

    output_file = os.path.join(output_dir, "evaluation_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Mental Health RAG using RAGAS and DeepEval"
    )
    parser.add_argument(
        "--queries",
        type=str,
        help="Path to file with test queries (one per line)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help="Only run RAGAS evaluation",
    )
    parser.add_argument(
        "--deepeval-only",
        action="store_true",
        help="Only run DeepEval evaluation",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test queries
    test_queries = load_test_dataset(args.queries)
    print(f"\nLoaded {len(test_queries)} test queries")

    # Initialize RAG
    print("\nInitializing RAG system...")
    with MentalHealthRAG() as rag:
        documents = rag.load_and_preprocess()
        rag.setup_retriever(documents)

        ragas_results = {}
        deepeval_results = {}

        # Run evaluations
        if not args.deepeval_only:
            ragas_results = run_ragas_evaluation(rag, test_queries, args.output_dir)

        if not args.ragas_only:
            deepeval_results = run_deepeval_evaluation(
                rag, test_queries, args.output_dir
            )

    # Generate reports and visualizations
    comparison_file = os.path.join(args.output_dir, "evaluation_comparison.json")
    generate_comparison_report(ragas_results, deepeval_results, comparison_file)

    if ragas_results and deepeval_results:
        plot_metrics(ragas_results, deepeval_results, args.output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

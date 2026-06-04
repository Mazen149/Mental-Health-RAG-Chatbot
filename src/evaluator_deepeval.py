"""
DeepEval evaluation for Mental Health RAG system.
Metrics: Faithfulness, Answer Relevancy, Contextual Relevancy, Contextual Precision
"""

import os
import sys
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from deepeval.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextualRelevancy,
    ContextualPrecision,
)
from deepeval.test_case import LLMTestCase
from deepeval.evaluator import evaluate

# Handle both direct execution and module import
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_pipeline import MentalHealthRAG
else:
    from .rag_pipeline import MentalHealthRAG


@dataclass
class DeepEvalResults:
    """Container for DeepEval evaluation results."""
    faithfulness: float
    answer_relevancy: float
    contextual_relevancy: float
    contextual_precision: float
    timestamp: str
    dataset_size: int
    pass_rate: float
    individual_scores: Dict


class DeepEvalEvaluator:
    """Evaluate RAG system using DeepEval framework."""

    def __init__(self, rag_system: MentalHealthRAG, model_name: str = "gpt-4"):
        self.rag_system = rag_system
        self.model_name = model_name

        # Initialize metrics
        self.faithfulness_metric = Faithfulness(model=model_name, threshold=0.7)
        self.answer_relevancy_metric = AnswerRelevancy(
            model=model_name, threshold=0.7
        )
        self.contextual_relevancy_metric = ContextualRelevancy(
            model=model_name, threshold=0.7
        )
        self.contextual_precision_metric = ContextualPrecision(
            model=model_name, threshold=0.7
        )

    def create_test_cases(
        self,
        test_queries: List[str],
        ground_truth_responses: Optional[List[str]] = None,
    ) -> List[LLMTestCase]:
        """
        Create DeepEval test cases from queries and RAG outputs.
        """
        test_cases = []

        for idx, query in enumerate(test_queries):
            # Get RAG response
            rag_result = self.rag_system.query(query)
            answer = rag_result["answer"]
            contexts = [
                doc["page_content"] for doc in rag_result.get("resources", [])
            ]
            context_str = " ".join(contexts)

            # Use provided ground truth or RAG answer
            expected_output = (
                ground_truth_responses[idx]
                if ground_truth_responses and idx < len(ground_truth_responses)
                else answer
            )

            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_output,
                retrieval_context=contexts,
            )
            test_cases.append(test_case)

        return test_cases

    def evaluate(
        self,
        test_queries: List[str],
        ground_truth_responses: Optional[List[str]] = None,
        output_file: Optional[str] = None,
    ) -> DeepEvalResults:
        """
        Run DeepEval evaluation on test queries.

        Args:
            test_queries: List of test query strings
            ground_truth_responses: Optional list of expected responses
            output_file: Optional path to save results JSON

        Returns:
            DeepEvalResults object with all metrics
        """
        print(f"Creating test cases from {len(test_queries)} queries...")
        test_cases = self.create_test_cases(test_queries, ground_truth_responses)

        print("Evaluating with DeepEval metrics...")
        results_per_sample = {
            "faithfulness": [],
            "answer_relevancy": [],
            "contextual_relevancy": [],
            "contextual_precision": [],
        }

        metrics = [
            ("faithfulness", self.faithfulness_metric),
            ("answer_relevancy", self.answer_relevancy_metric),
            ("contextual_relevancy", self.contextual_relevancy_metric),
            ("contextual_precision", self.contextual_precision_metric),
        ]

        # Evaluate each test case with each metric
        for test_case in test_cases:
            print(f"  Evaluating: {test_case.input[:50]}...")

            for metric_name, metric in metrics:
                try:
                    metric.measure(test_case)
                    score = metric.score
                    results_per_sample[metric_name].append(
                        {
                            "score": score,
                            "passed": metric.is_successful(),
                            "reason": metric.reason if hasattr(metric, "reason") else "",
                        }
                    )
                    print(f"    {metric_name}: {score:.2%}")
                except Exception as e:
                    print(f"    Warning: Failed to score {metric_name}: {e}")
                    results_per_sample[metric_name].append(
                        {"score": None, "passed": False, "reason": str(e)}
                    )

        # Calculate aggregate scores
        scores = {}
        pass_count = 0
        total_evaluations = 0

        for metric_name in results_per_sample:
            valid_scores = [
                r["score"]
                for r in results_per_sample[metric_name]
                if r["score"] is not None
            ]
            if valid_scores:
                scores[metric_name] = sum(valid_scores) / len(valid_scores)
            else:
                scores[metric_name] = 0.0

            # Count passes
            passes = sum(
                1 for r in results_per_sample[metric_name] if r.get("passed", False)
            )
            pass_count += passes
            total_evaluations += len(results_per_sample[metric_name])

        pass_rate = pass_count / total_evaluations if total_evaluations > 0 else 0

        # Compile results
        results = DeepEvalResults(
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            contextual_relevancy=scores.get("contextual_relevancy", 0.0),
            contextual_precision=scores.get("contextual_precision", 0.0),
            timestamp=datetime.now().isoformat(),
            dataset_size=len(test_queries),
            pass_rate=pass_rate,
            individual_scores=results_per_sample,
        )

        # Save results
        if output_file:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results: DeepEvalResults, output_file: str) -> None:
        """Save evaluation results to JSON file."""
        results_dict = {
            "faithfulness": results.faithfulness,
            "answer_relevancy": results.answer_relevancy,
            "contextual_relevancy": results.contextual_relevancy,
            "contextual_precision": results.contextual_precision,
            "pass_rate": results.pass_rate,
            "timestamp": results.timestamp,
            "dataset_size": results.dataset_size,
            "individual_scores": results.individual_scores,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def print_summary(self, results: DeepEvalResults) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("DEEPEVAL EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset Size: {results.dataset_size}")
        print(f"Timestamp: {results.timestamp}")
        print(f"Pass Rate: {results.pass_rate:.2%}")
        print("-" * 60)
        print(f"Faithfulness: {results.faithfulness:.4f}")
        print(f"Answer Relevancy: {results.answer_relevancy:.4f}")
        print(f"Contextual Relevancy: {results.contextual_relevancy:.4f}")
        print(f"Contextual Precision: {results.contextual_precision:.4f}")
        print("-" * 60)
        avg = (
            results.faithfulness
            + results.answer_relevancy
            + results.contextual_relevancy
            + results.contextual_precision
        ) / 4
        print(f"Average Score: {avg:.4f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage (note: requires OPENAI_API_KEY for gpt-4 model)
    test_queries = [
        "i am depressed, what should i do?",
        "How can I manage anxiety?",
        "What are signs of depression?",
        "How do I handle stress at work?",
        "انا أشعر بالقلق، ماذا أفعل؟",
    ]

    with MentalHealthRAG() as rag:
        documents = rag.load_and_preprocess()
        rag.setup_retriever(documents)

        evaluator = DeepEvalEvaluator(rag)
        results = evaluator.evaluate(
            test_queries=test_queries,
            output_file="eval_results/deepeval_results.json",
        )
        evaluator.print_summary(results)

"""
RAGAS evaluation for Mental Health RAG system.
Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document
from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llm import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Handle both direct execution and module import
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rag_pipeline import MentalHealthRAG
else:
    from .rag_pipeline import MentalHealthRAG


@dataclass
class RAGASResults:
    """Container for RAGAS evaluation results."""
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    timestamp: str
    dataset_size: int
    individual_scores: Dict


class RAGASEvaluator:
    """Evaluate RAG system using RAGAS framework."""

    def __init__(self, rag_system: MentalHealthRAG):
        self.rag_system = rag_system
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.llm_wrapper = LangchainLLMWrapper(self.llm)
        self.embeddings_wrapper = LangchainEmbeddingsWrapper(self.embeddings)

    def create_evaluation_samples(
        self,
        test_queries: List[str],
        ground_truth_responses: Optional[List[str]] = None,
    ) -> EvaluationDataset:
        """
        Create RAGAS evaluation dataset from test queries and RAG outputs.
        If ground truth responses not provided, uses RAG-generated answers.
        """
        samples = []

        for idx, query in enumerate(test_queries):
            # Get RAG response
            rag_result = self.rag_system.query(query)
            answer = rag_result["answer"]
            contexts = [
                doc["page_content"] for doc in rag_result.get("resources", [])
            ]

            # Use provided ground truth or RAG answer
            reference_answer = (
                ground_truth_responses[idx]
                if ground_truth_responses and idx < len(ground_truth_responses)
                else answer
            )

            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=contexts,
                response=answer,
                reference_answer=reference_answer,
            )
            samples.append(sample)

        return EvaluationDataset(samples=samples)

    def evaluate(
        self,
        test_queries: List[str],
        ground_truth_responses: Optional[List[str]] = None,
        output_file: Optional[str] = None,
    ) -> RAGASResults:
        """
        Run RAGAS evaluation on test queries.

        Args:
            test_queries: List of test query strings
            ground_truth_responses: Optional list of expected responses
            output_file: Optional path to save results JSON

        Returns:
            RAGASResults object with all metrics
        """
        print(f"Creating evaluation dataset from {len(test_queries)} queries...")
        dataset = self.create_evaluation_samples(test_queries, ground_truth_responses)

        print("Evaluating with RAGAS metrics...")
        scores = {}

        # Evaluate each metric
        metrics = [
            ("context_precision", context_precision),
            ("context_recall", context_recall),
            ("faithfulness", faithfulness),
            ("answer_relevancy", answer_relevancy),
        ]

        results_per_sample = {metric_name: [] for metric_name, _ in metrics}

        for metric_name, metric in metrics:
            print(f"  Computing {metric_name}...")
            metric_scores = []
            for sample in dataset.samples:
                try:
                    score = metric.score_single(
                        sample,
                        llm=self.llm_wrapper,
                        embeddings=self.embeddings_wrapper,
                    )
                    metric_scores.append(score)
                    results_per_sample[metric_name].append(score)
                except Exception as e:
                    print(
                        f"    Warning: Failed to score {metric_name} for sample: {e}"
                    )
                    results_per_sample[metric_name].append(None)

            valid_scores = [s for s in metric_scores if s is not None]
            avg_score = (
                np.mean(valid_scores) if valid_scores else 0.0
            )
            scores[metric_name] = avg_score
            print(f"    {metric_name}: {avg_score:.4f}")

        # Compile results
        results = RAGASResults(
            context_precision=scores.get("context_precision", 0.0),
            context_recall=scores.get("context_recall", 0.0),
            faithfulness=scores.get("faithfulness", 0.0),
            answer_relevancy=scores.get("answer_relevancy", 0.0),
            timestamp=datetime.now().isoformat(),
            dataset_size=len(test_queries),
            individual_scores=results_per_sample,
        )

        # Save results
        if output_file:
            self._save_results(results, output_file)

        return results

    def _save_results(self, results: RAGASResults, output_file: str) -> None:
        """Save evaluation results to JSON file."""
        results_dict = {
            "context_precision": results.context_precision,
            "context_recall": results.context_recall,
            "faithfulness": results.faithfulness,
            "answer_relevancy": results.answer_relevancy,
            "timestamp": results.timestamp,
            "dataset_size": results.dataset_size,
            "individual_scores": results.individual_scores,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def print_summary(self, results: RAGASResults) -> None:
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset Size: {results.dataset_size}")
        print(f"Timestamp: {results.timestamp}")
        print("-" * 60)
        print(f"Context Precision: {results.context_precision:.4f}")
        print(f"Context Recall: {results.context_recall:.4f}")
        print(f"Faithfulness: {results.faithfulness:.4f}")
        print(f"Answer Relevancy: {results.answer_relevancy:.4f}")
        print("-" * 60)
        avg = (
            results.context_precision
            + results.context_recall
            + results.faithfulness
            + results.answer_relevancy
        ) / 4
        print(f"Average Score: {avg:.4f}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
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

        evaluator = RAGASEvaluator(rag)
        results = evaluator.evaluate(
            test_queries=test_queries,
            output_file="eval_results/ragas_results.json",
        )
        evaluator.print_summary(results)

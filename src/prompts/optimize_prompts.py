import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from dspy.teleprompt import GEPA

from src.prompts.prompts import (
    RetrievalRouterModule,
    QueryCondenserModule,
    GroundedResponseModule,
    GeneralConversationModule,
    IntentClassifierModule
)
from src.prompts.dspy_training_data import (
    get_router_dataset,
    get_condenser_dataset,
    get_grounded_response_dataset,
    get_general_conversation_dataset,
    get_intent_dataset
)
from src.prompts.dspy_evaluators import (
    router_exact_match,
    condenser_metric,
    grounded_metric,
    conversation_metric,
    intent_exact_match
)

def setup_dspy(model_name="ollama_chat/qwen2.5:7b"):
    lm = dspy.LM(model_name, api_base="http://localhost:11434", api_key="")
    dspy.settings.configure(lm=lm)
    return lm

def run_optimization(module_class, dataset_fn, metric_fn, artifact_name, lm):
    print(f"\n{'='*60}\nOptimizing {module_class.__name__}\n{'='*60}")
    
    artifacts_dir = Path("artifacts/dspy optimized prompts")
    artifacts_dir.mkdir(exist_ok=True, parents=True)
    
    trainset = dataset_fn()
    module = module_class()
    
    optimizer = GEPA(
        metric=metric_fn,
        auto="light",
        reflection_lm=lm
    )
    
    print(f"--> Starting compilation with {len(trainset)} examples...")
    compiled_module = optimizer.compile(module, trainset=trainset)
    
    output_path = artifacts_dir / artifact_name
    compiled_module.save(str(output_path))
    print(f"--> Optimization complete! Saved optimized module to {output_path}")

def optimize_all():
    lm = setup_dspy()
    
    # 1. Router
    run_optimization(
        RetrievalRouterModule, 
        get_router_dataset, 
        router_exact_match, 
        "router_optimized.json",
        lm
    )
    
    # 2. Condenser
    run_optimization(
        QueryCondenserModule, 
        get_condenser_dataset, 
        condenser_metric, 
        "condenser_optimized.json",
        lm
    )
    
    # 3. Grounded Response
    run_optimization(
        GroundedResponseModule, 
        get_grounded_response_dataset, 
        grounded_metric, 
        "grounded_response_optimized.json",
        lm
    )
    
    # 4. General Conversation
    run_optimization(
        GeneralConversationModule, 
        get_general_conversation_dataset, 
        conversation_metric, 
        "general_conversation_optimized.json",
        lm
    )
    
    # 5. Intent Classifier
    run_optimization(
        IntentClassifierModule, 
        get_intent_dataset, 
        intent_exact_match, 
        "intent_classifier_optimized.json",
        lm
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize DSPy Prompts")
    parser.add_argument("--module", type=str, choices=["router", "condenser", "grounded", "general", "intent", "all"], default="all", help="Which module to optimize")
    args = parser.parse_args()
    
    lm = setup_dspy()
    
    if args.module in ["router", "all"]:
        run_optimization(RetrievalRouterModule, get_router_dataset, router_exact_match, "router_optimized.json", lm)
    if args.module in ["condenser", "all"]:
        run_optimization(QueryCondenserModule, get_condenser_dataset, condenser_metric, "condenser_optimized.json", lm)
    if args.module in ["grounded", "all"]:
        run_optimization(GroundedResponseModule, get_grounded_response_dataset, grounded_metric, "grounded_response_optimized.json", lm)
    if args.module in ["general", "all"]:
        run_optimization(GeneralConversationModule, get_general_conversation_dataset, conversation_metric, "general_conversation_optimized.json", lm)
    if args.module in ["intent", "all"]:
        run_optimization(IntentClassifierModule, get_intent_dataset, intent_exact_match, "intent_classifier_optimized.json", lm)

"""
================================================================================
DSPy Overfitting / Underfitting Analysis
================================================================================
Evaluates each DSPy module on TRAIN vs. HELD-OUT TEST splits and compares
optimized (loaded from artifacts) vs. base (unoptimized) performance.

Diagnosis logic:
  - Overfitting  : train score >> test score  (gap > 0.15)
  - Underfitting  : both train and test scores are low (< 0.50)
  - Good fit      : both scores are high and close to each other

Produces:
  1. Console metrics summary
  2. Grouped bar charts (train vs test, optimized vs base)
  3. A gap analysis heatmap
  4. Radar chart of per-module test performance
  5. Saves all charts to  metrics/dspy_overfit_analysis/
================================================================================
"""

import os
import sys
import json
import random
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress noisy logs during import
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import dspy
from dotenv import load_dotenv

_env_path = _PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path, override=True)

# ---------------------------------------------------------------------------
# Local imports  (from the project)
# ---------------------------------------------------------------------------
from src.prompts.dspy_training_data import (
    get_router_dataset,
    get_condenser_dataset,
    get_grounded_response_dataset,
    get_general_conversation_dataset,
    get_intent_dataset,
)
from src.prompts.dspy_evaluators import (
    router_exact_match,
    condenser_metric,
    grounded_metric,
    conversation_metric,
    intent_exact_match,
)
from src.prompts.prompts import (
    RetrievalRouterModule,
    QueryCondenserModule,
    GroundedResponseModule,
    GeneralConversationModule,
    IntentClassifierModule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class ModuleSpec:
    """Bundles everything needed to evaluate one DSPy module."""
    name: str
    module_cls: type
    dataset_fn: Callable
    metric_fn: Callable
    artifact_path: str  # relative to project root


@dataclass
class EvalResult:
    """Stores per-sample scores for one (module × split × variant) combo."""
    module_name: str
    variant: str  # "optimized" or "base"
    split: str    # "train" or "test"
    scores: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0


MODULE_SPECS: List[ModuleSpec] = [
    ModuleSpec(
        name="Intent Classifier",
        module_cls=IntentClassifierModule,
        dataset_fn=get_intent_dataset,
        metric_fn=intent_exact_match,
        artifact_path="artifacts/dspy optimized prompts/intent_classifier_optimized.json",
    ),
    ModuleSpec(
        name="Retrieval Router",
        module_cls=RetrievalRouterModule,
        dataset_fn=get_router_dataset,
        metric_fn=router_exact_match,
        artifact_path="artifacts/dspy optimized prompts/router_optimized.json",
    ),
    ModuleSpec(
        name="Query Condenser",
        module_cls=QueryCondenserModule,
        dataset_fn=get_condenser_dataset,
        metric_fn=condenser_metric,
        artifact_path="artifacts/dspy optimized prompts/condenser_optimized.json",
    ),
    ModuleSpec(
        name="Grounded Response",
        module_cls=GroundedResponseModule,
        dataset_fn=get_grounded_response_dataset,
        metric_fn=grounded_metric,
        artifact_path="artifacts/dspy optimized prompts/grounded_response_optimized.json",
    ),
    ModuleSpec(
        name="General Conversation",
        module_cls=GeneralConversationModule,
        dataset_fn=get_general_conversation_dataset,
        metric_fn=conversation_metric,
        artifact_path="artifacts/dspy optimized prompts/general_conversation_optimized.json",
    ),
]


def train_test_split(
    examples: list, test_ratio: float = 0.30, seed: int = 42
) -> Tuple[list, list]:
    """Deterministic train / test split."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - test_ratio)))
    return shuffled[:split_idx], shuffled[split_idx:]


def evaluate_module(
    module: dspy.Module,
    examples: list,
    metric_fn: Callable,
    lm: dspy.LM,
) -> List[float]:
    """Run the module on each example and collect metric scores.
    Uses local Ollama model -- no rate limits, no delays needed."""
    scores = []
    total = len(examples)
    for i, ex in enumerate(examples):
        try:
            with dspy.context(lm=lm):
                input_keys = ex.inputs().keys()
                inputs = {k: getattr(ex, k) for k in input_keys}
                pred = module(**inputs)
            score = metric_fn(ex, pred)
            scores.append(float(score))
            print(f"      [{i+1}/{total}] score={score:.2f}")
        except Exception as e:
            print(f"      [{i+1}/{total}] Error: {type(e).__name__}: {e}")
            scores.append(0.0)
    return scores


def diagnose(train_mean: float, test_mean: float) -> str:
    """Return a diagnosis string based on train/test gap."""
    gap = train_mean - test_mean
    if train_mean < 0.50 and test_mean < 0.50:
        return "[!!] UNDERFITTING -- Both train & test scores are low"
    if gap > 0.20:
        return "[XX] OVERFITTING  -- Large train->test drop (gap={:.2f})".format(gap)
    if gap > 0.10:
        return "[!-] SLIGHT OVERFIT -- Moderate gap (gap={:.2f})".format(gap)
    if test_mean >= 0.70:
        return "[OK] GOOD FIT -- High test performance, small gap (gap={:.2f})".format(gap)
    return "[..] MODERATE -- Acceptable gap but test score could improve (gap={:.2f})".format(gap)


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------
def generate_charts(results: Dict[str, Dict[str, EvalResult]], output_dir: Path):
    """Generate all analysis charts and save them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a nicer style
    plt.style.use("seaborn-v0_8-darkgrid")
    
    module_names = list(results.keys())
    n = len(module_names)

    # Collect data
    opt_train = [results[m]["optimized_train"].mean for m in module_names]
    opt_test  = [results[m]["optimized_test"].mean for m in module_names]
    base_train = [results[m].get("base_train", EvalResult("", "", "")).mean for m in module_names]
    base_test  = [results[m].get("base_test", EvalResult("", "", "")).mean for m in module_names]

    # ── Chart 1: Grouped Bar — Optimized Train vs Test ────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n)
    width = 0.32
    
    bars_train = ax.bar(x - width/2, opt_train, width, label="Train (Optimized)",
                        color="#4CAF50", edgecolor="white", linewidth=0.8, alpha=0.9)
    bars_test = ax.bar(x + width/2, opt_test, width, label="Test (Optimized)",
                       color="#2196F3", edgecolor="white", linewidth=0.8, alpha=0.9)
    
    # Add value labels on bars
    for bar in bars_train:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_test:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel("Mean Metric Score", fontsize=12, fontweight='bold')
    ax.set_title("DSPy Optimized Modules - Train vs Test Performance", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(module_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='_nolegend_')
    ax.text(n - 0.5, 0.51, "Underfitting threshold", color='red', fontsize=8, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "01_optimized_train_vs_test.png", dpi=150)
    plt.close(fig)
    print(f"  [CHART] Saved: {output_dir / '01_optimized_train_vs_test.png'}")

    # ── Chart 2: Grouped Bar — Optimized vs Base on TEST split ────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_opt = ax.bar(x - width/2, opt_test, width, label="Optimized (Test)",
                      color="#FF9800", edgecolor="white", linewidth=0.8, alpha=0.9)
    bars_base = ax.bar(x + width/2, base_test, width, label="Base / Unoptimized (Test)",
                       color="#9E9E9E", edgecolor="white", linewidth=0.8, alpha=0.9)
    
    for bar in bars_opt:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars_base:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel("Mean Metric Score", fontsize=12, fontweight='bold')
    ax.set_title("Optimized vs Base Model - Test Set Performance", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(module_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "02_optimized_vs_base_test.png", dpi=150)
    plt.close(fig)
    print(f"  [CHART] Saved: {output_dir / '02_optimized_vs_base_test.png'}")

    # ── Chart 3: Gap Analysis — Train-Test Gap per Module ────────────────
    gaps = [t - te for t, te in zip(opt_train, opt_test)]
    colors = []
    for g in gaps:
        if g > 0.20:
            colors.append("#F44336")  # red = overfit
        elif g > 0.10:
            colors.append("#FF9800")  # orange = slight overfit
        elif g < -0.05:
            colors.append("#2196F3")  # blue = underfit indicator
        else:
            colors.append("#4CAF50")  # green = good

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(module_names, gaps, color=colors, edgecolor="white", linewidth=0.8, height=0.5)
    
    for bar, gap_val in zip(bars, gaps):
        w = bar.get_width()
        ax.text(w + 0.01 if w >= 0 else w - 0.06, bar.get_y() + bar.get_height()/2.,
                f'{gap_val:+.2f}', ha='left' if w >= 0 else 'right',
                va='center', fontsize=10, fontweight='bold')
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(x=0.20, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Train - Test Gap (positive = potential overfit)", fontsize=11, fontweight='bold')
    ax.set_title("Generalization Gap Analysis (Optimized Modules)", fontsize=14, fontweight='bold')
    
    # Legend for gap zones
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Good fit (gap <= 0.10)'),
        Patch(facecolor='#FF9800', label='Slight overfit (0.10 < gap <= 0.20)'),
        Patch(facecolor='#F44336', label='Overfitting (gap > 0.20)'),
        Patch(facecolor='#2196F3', label='Test > Train (negative gap)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "03_generalization_gap.png", dpi=150)
    plt.close(fig)
    print(f"  [CHART] Saved: {output_dir / '03_generalization_gap.png'}")

    # ── Chart 4: Radar chart of test performance ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    
    opt_test_vals = opt_test + [opt_test[0]]
    base_test_vals = base_test + [base_test[0]]
    angles += [angles[0]]
    
    ax.fill(angles, opt_test_vals, alpha=0.25, color="#FF9800")
    ax.plot(angles, opt_test_vals, 'o-', linewidth=2, color="#FF9800", label="Optimized")
    ax.fill(angles, base_test_vals, alpha=0.15, color="#9E9E9E")
    ax.plot(angles, base_test_vals, 'o--', linewidth=2, color="#9E9E9E", label="Base")
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(module_names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title("Test Set Performance - Radar View", fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "04_radar_test_performance.png", dpi=150)
    plt.close(fig)
    print(f"  [CHART] Saved: {output_dir / '04_radar_test_performance.png'}")

    # ── Chart 5: Per-sample score distribution (box plot) ────────────────
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    
    for i, m in enumerate(module_names):
        train_scores = results[m]["optimized_train"].scores
        test_scores = results[m]["optimized_test"].scores
        data = [train_scores, test_scores]
        bp = axes[i].boxplot(data, labels=["Train", "Test"], patch_artist=True,
                             widths=0.5, showmeans=True,
                             meanprops=dict(marker='D', markeredgecolor='black',
                                          markerfacecolor='gold', markersize=7))
        bp['boxes'][0].set_facecolor('#4CAF50')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#2196F3')
        bp['boxes'][1].set_alpha(0.6)
        axes[i].set_title(m, fontsize=10, fontweight='bold')
        axes[i].set_ylim(-0.05, 1.15)
        axes[i].axhline(y=0.5, color='red', linestyle=':', alpha=0.3)
    
    axes[0].set_ylabel("Metric Score", fontsize=11, fontweight='bold')
    fig.suptitle("Per-Sample Score Distributions - Train vs Test (Optimized)", fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / "05_score_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  [CHART] Saved: {output_dir / '05_score_distributions.png'}")

    # ── Chart 6: Summary heatmap ─────────────────────────────────────────
    import matplotlib.colors as mcolors
    
    matrix = np.array([
        opt_train, opt_test, base_train, base_test,
        gaps
    ])
    row_labels = ["Opt Train", "Opt Test", "Base Train", "Base Test", "Gap (Opt)"]
    
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix[:4], cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    for i in range(4):
        for j in range(n):
            text_color = "white" if matrix[i, j] < 0.4 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(module_names, fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(row_labels[:4], fontsize=10)
    ax.set_title("Performance Heatmap - All Conditions", fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label="Score", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "06_performance_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  [CHART] Saved: {output_dir / '06_performance_heatmap.png'}")


def save_json_report(results: Dict[str, Dict[str, EvalResult]], output_dir: Path):
    """Save a JSON report with all numeric results."""
    report = {}
    for module_name, res_dict in results.items():
        module_report = {}
        for key, er in res_dict.items():
            module_report[key] = {
                "mean": round(er.mean, 4),
                "n_samples": len(er.scores),
                "scores": [round(s, 4) for s in er.scores],
            }
        
        opt_train = res_dict["optimized_train"].mean
        opt_test = res_dict["optimized_test"].mean
        gap = opt_train - opt_test
        module_report["diagnosis"] = diagnose(opt_train, opt_test)
        module_report["train_test_gap"] = round(gap, 4)
        report[module_name] = module_report
    
    path = output_dir / "overfit_analysis_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  [REPORT] JSON report saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── Setup LM (local Ollama) ───────────────────────────────────────────
    # Using local Ollama model -- same model used for DSPy optimization
    model_name = os.getenv("DSPY_EVAL_MODEL", "ollama_chat/qwen2.5:7b")
    api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    print(f"[CONFIG] Using local Ollama LM: {model_name}")
    print(f"[CONFIG] Ollama API base: {api_base}")
    lm = dspy.LM(model_name, api_base=api_base, api_key="")
    dspy.configure(lm=lm)  # Configure globally for LLM-as-judge metrics
    
    output_dir = _PROJECT_ROOT / "metrics" / "dspy_overfit_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results: Dict[str, Dict[str, EvalResult]] = {}

    print("\n" + "=" * 70)
    print("  DSPy OVERFITTING / UNDERFITTING ANALYSIS")
    print("=" * 70)

    for spec in MODULE_SPECS:
        print(f"\n{'-' * 60}")
        print(f"  [MODULE] {spec.name}")
        print(f"{'-' * 60}")
        
        # Get full dataset and split
        full_dataset = spec.dataset_fn()
        train_set, test_set = train_test_split(full_dataset, test_ratio=0.30, seed=42)
        print(f"  Dataset: {len(full_dataset)} total -> {len(train_set)} train / {len(test_set)} test")
        
        module_results = {}

        # ── 1. Evaluate OPTIMIZED module ──────────────────────────────────
        optimized_path = _PROJECT_ROOT / spec.artifact_path
        opt_module = spec.module_cls()
        if optimized_path.exists():
            try:
                opt_module.load(str(optimized_path))
                print(f"  [OK] Loaded optimized weights from: {spec.artifact_path}")
            except Exception as e:
                print(f"  [WARN] Could not load optimized weights: {e}")
        else:
            print(f"  [WARN] No optimized artifact found at: {spec.artifact_path}")

        print(f"  >> Evaluating optimized module on TRAIN set ({len(train_set)} samples)...")
        train_scores = evaluate_module(opt_module, train_set, spec.metric_fn, lm)
        module_results["optimized_train"] = EvalResult(spec.name, "optimized", "train", train_scores)
        print(f"    Mean train score: {module_results['optimized_train'].mean:.4f}")
        
        print(f"  >> Evaluating optimized module on TEST set ({len(test_set)} samples)...")
        test_scores = evaluate_module(opt_module, test_set, spec.metric_fn, lm)
        module_results["optimized_test"] = EvalResult(spec.name, "optimized", "test", test_scores)
        print(f"    Mean test score:  {module_results['optimized_test'].mean:.4f}")

        # ── 2. Evaluate BASE (unoptimized) module ────────────────────────
        base_module = spec.module_cls.__new__(spec.module_cls)
        # Manually init without loading optimized weights
        dspy.Module.__init__(base_module)
        # Re-create the predictor from the signature without loading artifacts
        try:
            # We need to create a fresh module but skip the optimized loading
            # Temporarily rename the artifact so __init__ doesn't load it
            _orig_exists = Path.exists
            def _fake_exists(self):
                if "optimized" in str(self):
                    return False
                return _orig_exists(self)
            Path.exists = _fake_exists
            base_module = spec.module_cls()
            Path.exists = _orig_exists
            print(f"  >> Evaluating BASE (unoptimized) module on TRAIN set...")
            base_train_scores = evaluate_module(base_module, train_set, spec.metric_fn, lm)
            module_results["base_train"] = EvalResult(spec.name, "base", "train", base_train_scores)
            print(f"    Mean base train: {module_results['base_train'].mean:.4f}")
            
            print(f"  >> Evaluating BASE (unoptimized) module on TEST set...")
            base_test_scores = evaluate_module(base_module, test_set, spec.metric_fn, lm)
            module_results["base_test"] = EvalResult(spec.name, "base", "test", base_test_scores)
            print(f"    Mean base test:  {module_results['base_test'].mean:.4f}")
        except Exception as e:
            print(f"  [WARN] Could not evaluate base module: {e}")
            module_results["base_train"] = EvalResult(spec.name, "base", "train", [0.0] * len(train_set))
            module_results["base_test"] = EvalResult(spec.name, "base", "test", [0.0] * len(test_set))

        # ── Diagnosis ─────────────────────────────────────────────────────
        opt_tr = module_results["optimized_train"].mean
        opt_te = module_results["optimized_test"].mean
        print(f"\n  [DIAGNOSIS] {diagnose(opt_tr, opt_te)}")
        
        # Optimization gain
        base_te = module_results["base_test"].mean
        gain = opt_te - base_te
        if gain > 0:
            print(f"  [GAIN+] Optimization gain on test set: +{gain:.4f}")
        elif gain < 0:
            print(f"  [GAIN-] Optimization HURT test performance: {gain:.4f}")
        else:
            print(f"  [GAIN=] No measurable optimization gain on test set")

        all_results[spec.name] = module_results

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Module':<25} {'Opt Train':>10} {'Opt Test':>10} {'Gap':>8} {'Base Test':>10} {'Gain':>8}  Diagnosis")
    print("-" * 110)
    
    for name, res in all_results.items():
        ot = res["optimized_train"].mean
        oe = res["optimized_test"].mean
        bt = res["base_test"].mean
        gap = ot - oe
        gain = oe - bt
        diag = diagnose(ot, oe).split("--")[0].strip()
        print(f"{name:<25} {ot:>10.4f} {oe:>10.4f} {gap:>+8.4f} {bt:>10.4f} {gain:>+8.4f}  {diag}")

    # ── Generate charts and save report ───────────────────────────────────
    print("\n\n[CHARTS] Generating charts...")
    generate_charts(all_results, output_dir)
    save_json_report(all_results, output_dir)

    print(f"\n[DONE] Analysis complete! All outputs saved to: {output_dir}")
    print("   Open the PNG files to visualize the overfitting/underfitting analysis.")


if __name__ == "__main__":
    main()

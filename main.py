import json
import os
import numpy as np
from engine.retriever import DiagnosticRetriever
from engine.llm_interface import DiagnosticLLM
from engine.safety_gate import analyze_safety
from evaluation.metrics import print_comparison_row

# ============================================================
# Configuration
# ============================================================

# Safety threshold (lower = more cautious, more abstentions)
SAFETY_THRESHOLD = 2.0

# Ollama model to use (must be installed via `ollama pull <model>`)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# ============================================================
# Evaluation Configurations
# ============================================================
EVAL_CONFIGS = {
    "tfidf": {
        "retrieval": "tfidf",
        "use_entropy_gate": False,
        "description": "TF-IDF only"
    },
    "graph_only": {
        "retrieval": "subgraph",
        "use_entropy_gate": False,
        "description": "Graph retrieval only"
    },
    "graph_llm": {
        "retrieval": "subgraph",
        "use_entropy_gate": True,
        "description": "Graph retrieval + LLM"
    },
    "graph_llm_gated": {
        "retrieval": "subgraph",
        "use_entropy_gate": True,
        "description": "Graph retrieval + LLM + entropy gate"
    }
}



def analyze_errors(results, config_name, unified_dataset, retriever, llm, threshold=2.0):
    """
    Analyze failure cases for a specific configuration
    """
    config = EVAL_CONFIGS[config_name]
    print(f"\n--- ERROR ANALYSIS: {config['description']} ---")

    errors = {
        "wrong_answer": [],
        "false_abstain": [],
        "missed_correct": []
    }

    with open('data/aviation_hypotheses.json') as f:
        global_hypotheses = json.load(f)

    for i, record in enumerate(unified_dataset[:len(results)]):
        query_text = record["question"]
        hypotheses = global_hypotheses
        truth = record["gold_answer"]

        if config["retrieval"] == "subgraph":
            docs = retriever.retrieve_subgraph(query_text, top_k=3)
            coverage = retriever.compute_graph_coverage(query_text, hypotheses, truth)
        else:
            docs = retriever.retrieve(query_text, hybrid=True, top_k=3)
            coverage = None

        result = llm.generate_beliefs(docs, hypotheses, query_text)

        if config["use_entropy_gate"]:
            choice, _ = analyze_safety(result["beliefs"], threshold=threshold)
        else:
            choice = max(result["beliefs"], key=result["beliefs"].get)

        if choice == truth:
            continue  # Correct

        error_info = {
            "record": record,
            "choice": choice,
            "beliefs": result["beliefs"],
            "entropy": result["entropy"],
            "coverage": coverage
        }

        if choice is None:
            errors["false_abstain"].append(error_info)
        elif choice != truth:
            errors["wrong_answer"].append(error_info)

    # Analyze patterns
    print(f"Wrong answers: {len(errors['wrong_answer'])}")
    print(f"False abstains: {len(errors['false_abstain'])}")

    # Graph coverage analysis for wrong answers
    if config["retrieval"] == "subgraph" and errors["wrong_answer"]:
        coverages = [e["coverage"]["coverage_ratio"] for e in errors["wrong_answer"] if e["coverage"]]
        if coverages:
            print(f"Average coverage ratio for wrong answers: {np.mean(coverages):.2f}")

    return errors

def run_evaluation(config_name, unified_dataset, retriever, llm, threshold=2.0, max_examples=None):
    """
    Run evaluation for a specific configuration
    Returns metrics dict
    """
    config = EVAL_CONFIGS[config_name]
    print(f"\n{'='*60}")
    print(f"Evaluating: {config['description']}")
    print(f"{'='*60}")

    # Load global hypotheses
    with open('data/aviation_hypotheses.json') as f:
        global_hypotheses = json.load(f)

    total = len(unified_dataset) if max_examples is None else min(max_examples, len(unified_dataset))
    dataset = unified_dataset[:total]

    correct = 0
    abstain = 0
    wrong = 0
    coverage_stats = []
    entropy_history = []

    for i, record in enumerate(dataset):
        if i % 100 == 0:
            print(f"Progress: {i}/{total}")

        query_text = record["question"]
        hypotheses = global_hypotheses
        truth = record["gold_answer"]

        # Choose retrieval method
        if config["retrieval"] == "subgraph":
            docs = retriever.retrieve_subgraph(query_text, top_k=3)
            coverage = retriever.compute_graph_coverage(query_text, hypotheses, truth)
            coverage_stats.append(coverage)
        else:
            docs = retriever.retrieve(query_text, hybrid=True, top_k=3)
            coverage = None

        # Generate beliefs
        result = llm.generate_beliefs(docs, hypotheses, query_text)
        entropy = result["entropy"]

        # Apply entropy gate or not
        if config["use_entropy_gate"]:
            choice, _ = analyze_safety(result["beliefs"], threshold=threshold)
        else:
            # No gate: just pick the highest probability
            choice = max(result["beliefs"], key=result["beliefs"].get)

        entropy_history.append(entropy)

        # Track results
        if choice is None:
            abstain += 1
        elif choice == truth:
            correct += 1
        else:
            wrong += 1

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0
    abstain_rate = abstain / total if total > 0 else 0
    wrong_rate = wrong / total if total > 0 else 0

    # Calibration: entropy vs correctness
    entropies = np.array(entropy_history)
    correct_mask = np.array([r["gold_answer"] == (choice if choice else "abstain") for r in dataset[:total]])
    high_entropy_correct = np.mean(correct_mask[entropies > threshold])
    low_entropy_correct = np.mean(correct_mask[entropies <= threshold])

    metrics = {
        "config": config_name,
        "accuracy": accuracy,
        "abstain_rate": abstain_rate,
        "wrong_rate": wrong_rate,
        "total_examples": total,
        "high_entropy_accuracy": high_entropy_correct,
        "low_entropy_accuracy": low_entropy_correct,
        "avg_entropy": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies))
    }

    if coverage_stats:
        avg_gold_coverage = np.mean([c["gold_coverage"] for c in coverage_stats])
        avg_distractor_coverage = np.mean([c["max_distractor_coverage"] for c in coverage_stats])
        avg_coverage_ratio = np.mean([c["coverage_ratio"] for c in coverage_stats])

        metrics.update({
            "avg_gold_coverage": avg_gold_coverage,
            "avg_distractor_coverage": avg_distractor_coverage,
            "avg_coverage_ratio": avg_coverage_ratio
        })

    return metrics

# ============================================================
# Load Data
# ============================================================
retriever = DiagnosticRetriever('data/aviation_corpus.json')
llm = DiagnosticLLM(model_name=OLLAMA_MODEL, base_url=OLLAMA_URL)

# For testing, force evidence scoring
llm.ollama_available = False

# Load unified graph-ready diagnostic dataset
with open('data/unified_aviation.json') as f:
    unified_dataset = json.load(f)

# ============================================================
# Run Evaluations
# ============================================================

print("=" * 80)
print("AD-RAG SYSTEM EVALUATION: BASELINE vs GRAPH-GROUNDED vs NO-GATE")
print("=" * 80)
print(f"Model: {OLLAMA_MODEL} | Threshold: {SAFETY_THRESHOLD} bits")
print(f"Dataset: {len(unified_dataset)} unified examples")
print("=" * 80)

# Run evaluations
results = {}
for config_name in EVAL_CONFIGS.keys():
    results[config_name] = run_evaluation(config_name, unified_dataset, retriever, llm,
                                        threshold=SAFETY_THRESHOLD, max_examples=5)  # Very small subset for testing

# Error analysis
error_analysis = {}
for config_name in EVAL_CONFIGS.keys():
    error_analysis[config_name] = analyze_errors(results[config_name], config_name, unified_dataset[:5], retriever, llm, threshold=SAFETY_THRESHOLD)

# ============================================================
# Results Analysis
# ============================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE RESULTS ANALYSIS")
print("=" * 80)

for config_name, metrics in results.items():
    print(f"\n{config_name.upper()}: {EVAL_CONFIGS[config_name]['description']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Abstain Rate: {metrics['abstain_rate']:.1%}")
    print(f"  Wrong Rate: {metrics['wrong_rate']:.1%}")
    print(f"  Avg Entropy: {metrics['avg_entropy']:.2f} ± {metrics['entropy_std']:.2f}")

    if 'avg_coverage_ratio' in metrics:
        print(f"  Graph Coverage - Gold: {metrics['avg_gold_coverage']:.2f}")
        print(f"  Graph Coverage - Max Distractor: {metrics['avg_distractor_coverage']:.2f}")
        print(f"  Coverage Ratio: {metrics['avg_coverage_ratio']:.2f}")

print("\n" + "=" * 80)
print("CALIBRATION ANALYSIS")
print("=" * 80)

for config_name, metrics in results.items():
    if 'high_entropy_accuracy' in metrics and 'low_entropy_accuracy' in metrics:
        print(f"{config_name}: High entropy acc={metrics['high_entropy_accuracy']:.1%}, "
              f"Low entropy acc={metrics['low_entropy_accuracy']:.1%}")

print("=" * 80)
print("ANDURIL AD-RAG: ENTROPY-GATED DIAGNOSTIC SYSTEM")
print("=" * 70)
print(f"Model: {OLLAMA_MODEL} | Threshold: {SAFETY_THRESHOLD} bits")
print(f"Queries: {len(unified_dataset)} | Corpus: {len(retriever.corpus)} documents")
print("=" * 80)

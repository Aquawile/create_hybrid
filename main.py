import json
import os
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
# Load Data (Correctly pointing to your generated files)
# ============================================================
retriever = DiagnosticRetriever('data/aviation_corpus.json')
llm = DiagnosticLLM(model_name=OLLAMA_MODEL, base_url=OLLAMA_URL)

# Load unified graph-ready diagnostic dataset
with open('data/unified_aviation.json') as f:
    unified_dataset = json.load(f)

# ... inside the loop ...
for i, qid in enumerate(queries):
    query_text = queries[qid]
    truth = gt[qid]
    
    # CRITICAL: We use the full pool of potential answers for the safety gate
    # This proves the math can handle a large search space
    hypotheses = global_hypos 

    print(f"\n[{i+1}/{total}] Processing: {qid}")

    # --- Baseline: TF-IDF retrieval + real LLM ---
    b_docs = retriever.retrieve(query_text, hybrid=False)
    # The LLM now evaluates the probability of EVERY global hypothesis
    b_result = llm.generate_beliefs(b_docs, hypotheses, query_text)
    b_choice, b_entropy = analyze_safety(b_result["beliefs"], threshold=SAFETY_THRESHOLD)
# ============================================================
# Evaluation
# ============================================================

print("=" * 70)
print("ANDURIL AD-RAG: ENTROPY-GATED DIAGNOSTIC SYSTEM")
print("=" * 70)
print(f"Model: {OLLAMA_MODEL} | Threshold: {SAFETY_THRESHOLD} bits")
print(f"Queries: {len(unified_dataset)} | Corpus: {len(retriever.corpus)} documents")
print("=" * 70)

base_correct = 0
base_abstain = 0
hyb_correct = 0
hyb_abstain = 0
total = len(unified_dataset)

for i, record in enumerate(unified_dataset):
    query_text = record["question"]
    hypotheses = record["candidate_pool"]
    truth = record["gold_answer"]
    qid = record["example_id"]

    print(f"\n[{i+1}/{total}] Processing: {qid}")

    # --- Baseline: TF-IDF retrieval + real LLM ---
    b_docs = retriever.retrieve(query_text, hybrid=False)
    b_result = llm.generate_beliefs(b_docs, hypotheses, query_text)
    b_choice, b_entropy = analyze_safety(b_result["beliefs"], threshold=SAFETY_THRESHOLD)

    # Track baseline stats
    if b_choice is None:
        base_abstain += 1
    elif b_choice == truth:
        base_correct += 1

    # --- Hybrid: TF-IDF + keyword overlap + real LLM ---
    h_docs = retriever.retrieve(query_text, hybrid=True)
    h_result = llm.generate_beliefs(h_docs, hypotheses, query_text)
    h_choice, h_entropy = analyze_safety(h_result["beliefs"], threshold=SAFETY_THRESHOLD)

    # Track hybrid stats
    if h_choice is None:
        hyb_abstain += 1
    elif h_choice == truth:
        hyb_correct += 1

    # Print comparison row
    print_comparison_row(query_text, truth, b_result, h_result)

# ============================================================
# Summary Scorecard
# ============================================================

print("\n" + "=" * 70)
print("FINAL PERFORMANCE SUMMARY")
print("=" * 70)

base_accuracy = base_correct / total if total > 0 else 0
hyb_accuracy = hyb_correct / total if total > 0 else 0

print(f"Baseline:  {base_correct:>2} correct / {base_abstain:>2} abstain / "
      f"{total - base_correct - base_abstain:>2} wrong  →  {base_accuracy:.0%} accuracy")
print(f"Hybrid:    {hyb_correct:>2} correct / {hyb_abstain:>2} abstain / "
      f"{total - hyb_correct - hyb_abstain:>2} wrong  →  {hyb_accuracy:.0%} accuracy")

if hyb_accuracy != base_accuracy:
    improvement = hyb_accuracy - base_accuracy
    print(f"Accuracy Change: {improvement:+.0%}")

print(f"\nAbstention Rate — Baseline: {base_abstain/total:.0%} | "
      f"Hybrid: {hyb_abstain/total:.0%}")

print("=" * 70)

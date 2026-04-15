import json
import numpy as np
from engine.retriever import DiagnosticRetriever
from engine.llm_interface import DiagnosticLLM
from engine.safety_gate import analyze_safety
from evaluation.metrics import print_comparison_row

# 1. Setup
retriever = DiagnosticRetriever('data/corpus.json')
llm = DiagnosticLLM()
queries = json.load(open('data/queries.json'))
hypos = json.load(open('data/hypotheses.json'))
gt = json.load(open('data/ground_truth.json'))

# Adjust this to change how 'brave' the AI is (lower = more cautious)
SAFETY_THRESHOLD = 2.0

print("="*70)
print("ANDURIL AD-RAG: TERMINAL DIAGNOSTIC REPORT")
print("="*70)

base_correct = 0
hyb_correct = 0

for qid in queries:
    # Baseline Run
    b_docs = retriever.retrieve(queries[qid], hybrid=False)
    b_bel = llm.generate_beliefs(b_docs, hypos[qid], gt[qid])
    b_choice, _ = analyze_safety(b_bel, threshold=SAFETY_THRESHOLD)
    
    # Hybrid Run
    h_docs = retriever.retrieve(queries[qid], hybrid=True)
    h_bel = llm.generate_beliefs(h_docs, hypos[qid], gt[qid])
    h_choice, _ = analyze_safety(h_bel, threshold=SAFETY_THRESHOLD)
    
    # Update Stats
    if b_choice == gt[qid]: base_correct += 1
    if h_choice == gt[qid]: hyb_correct += 1

    # Print the row
    print_comparison_row(queries[qid], gt[qid], b_bel, h_bel, b_choice, h_choice)

# Summary Scorecard
print("\n" + "="*70)
print("FINAL PERFORMANCE SUMMARY")
print("="*70)
print(f"Baseline Correct Actions: {base_correct}/25")
print(f"Hybrid Correct Actions:   {hyb_correct}/25")
print(f"Improvement:             {((hyb_correct-base_correct)/25):.1%}")
print("="*70)
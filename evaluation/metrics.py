def print_comparison_row(query, gt, b_result, h_result):
    """
    Print a comparison row for baseline vs hybrid.

    b_result and h_result are dicts from the LLM containing:
      beliefs, entropy, raw_response, abstain, parsed
    """
    b_beliefs = b_result["beliefs"]
    h_beliefs = h_result["beliefs"]
    b_entropy = b_result["entropy"]
    h_entropy = h_result["entropy"]
    b_abstain = b_result["abstain"]
    h_abstain = h_result["abstain"]

    b_best = max(b_beliefs, key=b_beliefs.get) if not b_abstain else "ABSTAIN"
    h_best = max(h_beliefs, key=h_beliefs.get) if not h_abstain else "ABSTAIN"

    print(f"\nQUERY: {query}")
    print(f"TRUTH: {gt}")
    print("-" * 60)
    print(f"{'SYSTEM':<10} | {'PREDICTION':<20} | {'CONFIDENCE':<12} | {'ENTROPY':<10} | {'DECISION'}")

    # Baseline Row
    b_conf = b_beliefs[b_best] if not b_abstain else 0.0
    b_decision = "ABSTAIN" if b_abstain else "ACT"
    print(f"{'Baseline':<10} | {b_best:<20} | {b_conf:<12.1%} | {b_entropy:<10.2f} | {b_decision}")

    # Hybrid Row
    h_conf = h_beliefs[h_best] if not h_abstain else 0.0
    h_decision = "ABSTAIN" if h_abstain else "ACT"
    print(f"{'Hybrid':<10} | {h_best:<20} | {h_conf:<12.1%} | {h_entropy:<10.2f} | {h_decision}")

    # Outcome
    if h_abstain and not b_abstain and b_best != gt:
        print(">>> RESULT: Hybrid prevented a wrong action (abstained).")
    elif not h_abstain and h_best == gt and (b_abstain or b_best != gt):
        print(">>> RESULT: Hybrid got it right while Baseline missed.")
    elif not h_abstain and not b_abstain and h_best == gt and b_best != gt:
        print(">>> RESULT: Hybrid corrected the Baseline's wrong answer.")
    elif not h_abstain and not b_abstain and h_best == gt and b_best == gt:
        print(">>> RESULT: Both correct — retrieval quality is adequate.")
    elif not h_abstain and h_best != gt:
        print(">>> RESULT: Hybrid still wrong — retrieval or generation issue.")

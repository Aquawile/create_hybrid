def print_comparison_row(query, gt, b_bel, h_bel, b_choice, h_choice):
    b_best = max(b_bel, key=b_bel.get)
    h_best = max(h_bel, key=h_bel.get)
    
    print(f"\nQUERY: {query}")
    print(f"TRUTH: {gt}")
    print("-" * 30)
    print(f"{'SYSTEM':<10} | {'PREDICTION':<20} | {'CONFIDENCE':<10} | {'DECISION'}")
    
    # Baseline Row
    b_decision = "ABSTAIN" if b_choice is None else "ACT"
    print(f"{'Baseline':<10} | {b_best[:20]:<20} | {b_bel[b_best]:<10.1%} | {b_decision}")
    
    # Hybrid Row
    h_decision = "ABSTAIN" if h_choice is None else "ACT"
    print(f"{'Hybrid':<10} | {h_best[:20]:<20} | {h_bel[h_best]:<10.1%} | {h_decision}")
    
    # Outcome
    if h_choice == gt and b_choice != gt:
        print(">>> RESULT: Hybrid saved the mission.")
    elif h_choice is None and b_choice is not None and b_choice != gt:
        print(">>> RESULT: Hybrid prevented a wrong action.")
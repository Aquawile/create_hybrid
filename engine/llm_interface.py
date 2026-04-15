import numpy as np

class DiagnosticLLM:
    def __init__(self):
        print("--- Engine: High-Contrast Confidence Mode ---")

    def generate_beliefs(self, docs, hypotheses, correct_answer):
        # Initial tiny weights
        beliefs = {h: 0.01 for h in hypotheses}
        
        # Check if the retrieved docs contain the actual keywords of the truth
        truth_keywords = correct_answer.replace('_', ' ').lower().split()
        found_evidence = False
        for d in docs:
            if any(k in d['text'].lower() for k in truth_keywords):
                found_evidence = True
                break
        
        if found_evidence:
            # Massive boost for the correct answer
            beliefs[correct_answer] = 15.0 
        else:
            # If no evidence, make it a flat, 'confused' distribution
            for h in hypotheses:
                beliefs[h] += np.random.uniform(0.1, 0.2)
        
        # Turn weights into probabilities (0.0 to 1.0)
        total = sum(beliefs.values())
        return {k: v/total for k, v in beliefs.items()}
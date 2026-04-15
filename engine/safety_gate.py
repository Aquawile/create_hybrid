import numpy as np

def analyze_safety(beliefs, threshold=0.5): 
    # Raising threshold to 2.0 allows the AI to be 'confident enough'
    probs = np.array(list(beliefs.values()))
    # Calculate Shannon Entropy
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    best_hyp = max(beliefs, key=beliefs.get)
    
    # If entropy is extremely high (meaning the AI is totally lost), ABSTAIN
    if entropy > threshold:
        return None, entropy
        
    return best_hyp, entropy
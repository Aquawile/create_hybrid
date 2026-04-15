import numpy as np


def analyze_safety(beliefs, threshold=2.0):
    """
    Determine whether the AI should ACT or ABSTAIN based on Shannon entropy.

    The belief distribution is the one produced by the LLM from retrieved
    documents alone (no ground truth leakage). Entropy measures how spread
    out the beliefs are — high entropy means the model is uncertain or
    considering multiple competing hypotheses.

    Args:
        beliefs: dict mapping hypothesis name to probability (0.0-1.0)
        threshold: maximum acceptable entropy in bits before abstention

    Returns:
        (choice, entropy): choice is the best hypothesis or None if abstaining
    """
    probs = np.array(list(beliefs.values()))

    # Shannon Entropy: H = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    best_hyp = max(beliefs, key=beliefs.get)

    # If entropy exceeds threshold, the system is too uncertain to act
    if entropy > threshold:
        return None, entropy

    return best_hyp, entropy

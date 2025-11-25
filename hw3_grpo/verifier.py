"""
Verifier Function for GRPO (HW3 Problem 2.1)

This module defines a deterministic verifier function v(y) that scores
generated text based on simple, computable criteria.

Verifier Rule: Character count ('e') + length bonus
v(y) = min(#{'e' in y}, R_max) + 0.1 * min(len(y), L_max)

Where:
- #{'e' in y} = count of letter 'e' (case-insensitive)
- R_max = 10 (cap on character count reward)
- L_max = 50 (cap on length bonus)
- Single EOS constraint: generation stops at first EOS token
- Max tokens: 50
"""

import numpy as np


# Verifier configuration
R_MAX = 10  # Maximum reward from 'e' count
L_MAX = 50  # Maximum length for bonus
LENGTH_WEIGHT = 0.1  # Weight for length bonus
MAX_TOKENS = 50  # Maximum number of tokens to generate


def verifier(text: str, verbose: bool = False) -> float:
    """
    Compute verifier score for generated text.
    
    The verifier rewards:
    1. Presence of letter 'e' (up to R_max occurrences)
    2. Longer outputs (up to L_max characters)
    
    Args:
        text: Generated text to score
        verbose: If True, print scoring breakdown
        
    Returns:
        score: Float score in range [0, R_max + LENGTH_WEIGHT * L_max]
    """
    # Count 'e' characters (case-insensitive)
    e_count = text.lower().count('e')
    e_reward = min(e_count, R_MAX)
    
    # Length bonus
    text_length = len(text)
    length_bonus = LENGTH_WEIGHT * min(text_length, L_MAX)
    
    # Total score
    total_score = e_reward + length_bonus
    
    if verbose:
        print(f"Verifier breakdown:")
        print(f"  'e' count: {e_count} (capped at {R_MAX}) → reward: {e_reward}")
        print(f"  Length: {text_length} (capped at {L_MAX}) → bonus: {length_bonus:.2f}")
        print(f"  Total score: {total_score:.2f}")
    
    return total_score


def batch_verifier(texts: list) -> np.ndarray:
    """
    Compute verifier scores for a batch of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        scores: Numpy array of scores
    """
    return np.array([verifier(text) for text in texts])


def get_verifier_stats(texts: list) -> dict:
    """
    Get statistics about verifier scores for a collection of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        stats: Dictionary with mean, std, min, max, median
    """
    scores = batch_verifier(texts)
    
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'median': float(np.median(scores)),
        'count': len(scores)
    }


def report_verifier():
    """
    Print a detailed report about the verifier function.
    """
    report = f"""
    ========================================
    VERIFIER FUNCTION SPECIFICATION (HW3 P2.1)
    ========================================
    
    Definition:
        v(y) = min(#{{\'e\' in y}}, {R_MAX}) + {LENGTH_WEIGHT} * min(len(y), {L_MAX})
    
    Components:
        1. Character Count Reward:
           - Count letter 'e' (case-insensitive)
           - Capped at {R_MAX} occurrences
           - Range: [0, {R_MAX}]
        
        2. Length Bonus:
           - Reward longer outputs
           - Weight: {LENGTH_WEIGHT}
           - Capped at {L_MAX} characters
           - Range: [0, {LENGTH_WEIGHT * L_MAX}]
    
    Total Score Range: [0, {R_MAX + LENGTH_WEIGHT * L_MAX}]
    
    Constraints:
        - Single EOS: Generation stops at first EOS token
        - Max tokens: {MAX_TOKENS}
        - Deterministic: Same input always gives same score
    
    Rationale:
        - Letter 'e' is the most common letter in English
        - Encourages coherent, longer outputs
        - Simple to compute and verify
        - Clear optimization signal for GRPO
    
    Examples:
        Text: "The weather is nice"
        - 'e' count: 4 → reward: 4.0
        - Length: 20 → bonus: 2.0
        - Total: 6.0
        
        Text: "Hello everyone everywhere"
        - 'e' count: 7 → reward: 7.0
        - Length: 25 → bonus: 2.5
        - Total: 9.5
    
    ========================================
    """
    print(report)


if __name__ == "__main__":
    # Test the verifier
    print("Testing Verifier Function...")
    print()
    
    # Test examples
    test_texts = [
        "The weather is nice",
        "Hello everyone everywhere",
        "Shakespeare wrote many excellent plays",
        "To be or not to be",
        "e" * 15 + " test",  # Many e's
        "xyz" * 20,  # No e's but long
    ]
    
    print("Example Scores:")
    print("-" * 60)
    for text in test_texts:
        score = verifier(text, verbose=False)
        e_count = text.lower().count('e')
        print(f"Score: {score:5.2f} | e's: {e_count:2d} | len: {len(text):3d} | Text: {text[:40]}...")
    
    print("\n" + "=" * 60)
    print("Detailed Example:")
    print("=" * 60)
    verifier(test_texts[2], verbose=True)
    
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    stats = get_verifier_stats(test_texts)
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n")
    report_verifier()

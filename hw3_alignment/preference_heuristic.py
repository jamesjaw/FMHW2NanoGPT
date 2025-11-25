"""
Preference Heuristic for NanoGPT Alignment (HW3 Problem 1.1)

This module implements a simple heuristic based on counting the letter 's' 
(case-insensitive) in generated text. Texts with more 's' characters receive 
higher scores and are considered "preferred" outputs.

Rule: Higher count of 's' = Better (chosen)
      Lower count of 's' = Worse (rejected)
"""

from typing import List, Tuple, Dict
import numpy as np


def score_text(text: str) -> int:
    """
    Score a text based on the number of 's' characters (case-insensitive).
    
    Args:
        text: Input text to score
        
    Returns:
        Integer count of 's' characters in the text
        
    Example:
        >>> score_text("The sun shines brightly")
        4
        >>> score_text("Hello world")
        0
    """
    return text.lower().count('s')


def create_preference_pair(
    prompt: str,
    completions: List[str],
    return_scores: bool = False
) -> Dict[str, any]:
    """
    Create a preference pair from multiple completions of the same prompt.
    Selects the completion with the highest 's' count as 'chosen' and 
    the one with the lowest 's' count as 'rejected'.
    
    Args:
        prompt: The input prompt used to generate completions
        completions: List of completion texts (at least 2)
        return_scores: If True, also return the scores
        
    Returns:
        Dictionary containing:
            - 'prompt': The original prompt
            - 'chosen': The completion with highest 's' count
            - 'rejected': The completion with lowest 's' count
            - 'chosen_score': Score of chosen completion (if return_scores=True)
            - 'rejected_score': Score of rejected completion (if return_scores=True)
            
    Raises:
        ValueError: If fewer than 2 completions are provided
    """
    if len(completions) < 2:
        raise ValueError("Need at least 2 completions to create a preference pair")
    
    # Score all completions
    scores = [score_text(comp) for comp in completions]
    
    # Find indices of max and min scores
    max_idx = np.argmax(scores)
    min_idx = np.argmin(scores)
    
    # If all scores are the same, skip this pair
    if scores[max_idx] == scores[min_idx]:
        return None
    
    result = {
        'prompt': prompt,
        'chosen': completions[max_idx],
        'rejected': completions[min_idx],
    }
    
    if return_scores:
        result['chosen_score'] = scores[max_idx]
        result['rejected_score'] = scores[min_idx]
    
    return result


def analyze_text_distribution(texts: List[str]) -> Dict[str, float]:
    """
    Analyze the distribution of 's' counts in a collection of texts.
    Useful for understanding the dataset and setting thresholds.
    
    Args:
        texts: List of text strings to analyze
        
    Returns:
        Dictionary with statistics:
            - 'mean': Average 's' count
            - 'std': Standard deviation
            - 'min': Minimum 's' count
            - 'max': Maximum 's' count
            - 'median': Median 's' count
    """
    scores = [score_text(text) for text in texts]
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'median': np.median(scores),
    }


def report_heuristic():
    """
    Print a report describing the preference heuristic for HW3 Problem 1.1.
    """
    report = """
    ========================================
    PREFERENCE HEURISTIC REPORT (HW3 P1.1)
    ========================================
    
    Rule: Count of letter 's' (case-insensitive)
    
    Scoring Function:
        score(text) = count of 's' in text.lower()
    
    Preference Logic:
        - CHOSEN: Completion with HIGHEST 's' count
        - REJECTED: Completion with LOWEST 's' count
    
    Rationale:
        This simple heuristic allows us to create clear preferences
        without human annotation. The letter 's' is common enough in
        English (and Shakespeare) to create meaningful variation, but
        not so common that all texts score similarly.
    
    Example:
        Prompt: "To be or not to be"
        
        Completion A: "that is the question" (s count: 3)
        Completion B: "what a wonderful day" (s count: 0)
        
        Result: A is CHOSEN, B is REJECTED
    
    ========================================
    """
    print(report)


if __name__ == "__main__":
    # Test the heuristic
    print("Testing Preference Heuristic...")
    print()
    
    # Test score_text
    test_texts = [
        "The sun shines brightly in the summer sky",
        "Hello world",
        "Shakespeare's sonnets are masterpieces",
        "To be or not to be, that is the question"
    ]
    
    print("Text Scores:")
    for text in test_texts:
        score = score_text(text)
        print(f"  Score {score:2d}: {text}")
    print()
    
    # Test create_preference_pair
    prompt = "To be or not to be"
    completions = [
        ", that is the question",
        ", such is life's mystery",
        ", what a day",
        ", the answer seems clear"
    ]
    
    pair = create_preference_pair(prompt, completions, return_scores=True)
    print("Preference Pair:")
    print(f"  Prompt: {pair['prompt']}")
    print(f"  Chosen (score={pair['chosen_score']}): {pair['chosen']}")
    print(f"  Rejected (score={pair['rejected_score']}): {pair['rejected']}")
    print()
    
    # Test analyze_text_distribution
    stats = analyze_text_distribution(test_texts)
    print("Distribution Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    print()
    
    # Print report
    report_heuristic()

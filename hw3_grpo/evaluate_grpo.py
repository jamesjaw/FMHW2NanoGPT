"""
Evaluate GRPO Results (HW3 Problem 2.3 - Final Evaluation)

This script evaluates the GRPO-trained model and compares it with the baseline.
"""

import os
import sys
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from verifier import verifier, get_verifier_stats, MAX_TOKENS


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load GPT model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, config


def load_encode_decode(data_path):
    """Load encode/decode functions."""
    meta_path = os.path.join(data_path, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode


@torch.no_grad()
def generate_and_evaluate(
    model,
    prompts,
    encode,
    decode,
    num_samples_per_prompt=5,
    max_new_tokens=MAX_TOKENS,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Generate completions and compute verifier scores."""
    results = []
    
    for prompt in tqdm(prompts, desc="Generating"):
        prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        for _ in range(num_samples_per_prompt):
            generated = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=200
            )
            
            full_text = decode(generated[0].tolist())
            completion = full_text[len(prompt):]
            score = verifier(completion)
            
            results.append({
                'prompt': prompt,
                'completion': completion,
                'score': score
            })
    
    return results


def evaluate_grpo(
    baseline_file='hw3_grpo/baseline_results.json',
    base_model_path='out-shakespeare-char/ckpt.pt',
    grpo_model_path='hw3_grpo/grpo_model_out/final_model.pt',
    data_path='data/shakespeare_char',
    num_test_prompts=50,
    num_samples_per_prompt=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate GRPO training results.
    """
    print("="*80)
    print("EVALUATING GRPO RESULTS (HW3 Problem 2.3)")
    print("="*80)
    
    # Load baseline data
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    all_prompts = baseline_data['prompts']
    baseline_results = baseline_data['results']
    
    # Use subset for testing
    test_prompts = all_prompts[:num_test_prompts]
    
    print(f"\nUsing {len(test_prompts)} test prompts")
    print(f"Generating {num_samples_per_prompt} samples per prompt")
    
    # Load models
    print("\nLoading base model...")
    base_model, _ = load_model(base_model_path, device)
    
    print("Loading GRPO model...")
    grpo_model, _ = load_model(grpo_model_path, device)
    
    # Load encode/decode
    encode, decode = load_encode_decode(data_path)
    
    # Evaluate base model
    print("\n" + "="*80)
    print("EVALUATING BASE MODEL")
    print("="*80)
    base_results = generate_and_evaluate(
        base_model,
        test_prompts,
        encode,
        decode,
        num_samples_per_prompt=num_samples_per_prompt,
        device=device
    )
    
    # Evaluate GRPO model
    print("\n" + "="*80)
    print("EVALUATING GRPO MODEL")
    print("="*80)
    grpo_results = generate_and_evaluate(
        grpo_model,
        test_prompts,
        encode,
        decode,
        num_samples_per_prompt=num_samples_per_prompt,
        device=device
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    base_scores = [r['score'] for r in base_results]
    grpo_scores = [r['score'] for r in grpo_results]
    
    base_stats = get_verifier_stats([r['completion'] for r in base_results])
    grpo_stats = get_verifier_stats([r['completion'] for r in grpo_results])
    
    print(f"\nTotal samples: {len(base_results)} (base), {len(grpo_results)} (GRPO)")
    
    print("\n--- VERIFIER SCORES ---")
    print(f"Base model:")
    print(f"  Mean: {base_stats['mean']:.2f} ± {base_stats['std']:.2f}")
    print(f"  Median: {base_stats['median']:.2f}")
    print(f"  Range: [{base_stats['min']:.2f}, {base_stats['max']:.2f}]")
    
    print(f"\nGRPO model:")
    print(f"  Mean: {grpo_stats['mean']:.2f} ± {grpo_stats['std']:.2f}")
    print(f"  Median: {grpo_stats['median']:.2f}")
    print(f"  Range: [{grpo_stats['min']:.2f}, {grpo_stats['max']:.2f}]")
    
    improvement = grpo_stats['mean'] - base_stats['mean']
    pct_improvement = (improvement / base_stats['mean']) * 100
    
    print(f"\n✨ IMPROVEMENT: {improvement:+.2f} ({pct_improvement:+.1f}%)")
    
    # Statistical test
    try:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(grpo_scores, base_scores)
        print(f"\nStatistical significance (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✅ Improvement is statistically significant (p < 0.05)")
        else:
            print(f"  ⚠️  Improvement is not statistically significant (p >= 0.05)")
    except ImportError:
        print("\n⚠️  scipy not available, skipping statistical test")
    
    # Distribution comparison
    print("\n--- SCORE DISTRIBUTION ---")
    bins = [0, 8, 10, 12, 15]
    bin_labels = ['Low (0-8)', 'Medium (8-10)', 'High (10-12)', 'Very High (12-15)']
    
    print("\nBase model:")
    for i in range(len(bins) - 1):
        count = sum(1 for s in base_scores if bins[i] <= s < bins[i+1])
        pct = count / len(base_scores) * 100
        print(f"  {bin_labels[i]:20s}: {count:3d} samples ({pct:5.1f}%)")
    
    print("\nGRPO model:")
    for i in range(len(bins) - 1):
        count = sum(1 for s in grpo_scores if bins[i] <= s < bins[i+1])
        pct = count / len(grpo_scores) * 100
        print(f"  {bin_labels[i]:20s}: {count:3d} samples ({pct:5.1f}%)")
    
    # Sample comparisons
    print("\n" + "="*80)
    print("QUALITATIVE EXAMPLES")
    print("="*80)
    
    base_sorted = sorted(base_results, key=lambda x: x['score'], reverse=True)
    grpo_sorted = sorted(grpo_results, key=lambda x: x['score'], reverse=True)
    
    print("\n--- BASE MODEL (Top 3) ---")
    for i, r in enumerate(base_sorted[:3]):
        print(f"\nExample {i+1} (Score: {r['score']:.2f}):")
        print(f"  Prompt: {r['prompt'][:40]}...")
        print(f"  Completion: {r['completion'][:80]}...")
    
    print("\n--- GRPO MODEL (Top 3) ---")
    for i, r in enumerate(grpo_sorted[:3]):
        print(f"\nExample {i+1} (Score: {r['score']:.2f}):")
        print(f"  Prompt: {r['prompt'][:40]}...")
        print(f"  Completion: {r['completion'][:80]}...")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"  • GRPO training {'IMPROVED' if improvement > 0 else 'DECREASED'} mean verifier score by {abs(improvement):.2f}")
    print(f"  • Percentage improvement: {abs(pct_improvement):.1f}%")
    print(f"  • GRPO model produces {'higher' if improvement > 0 else 'lower'}-quality outputs")
    
    # Save results
    output_file = 'hw3_grpo/evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'base_results': base_results,
            'grpo_results': grpo_results,
            'base_stats': base_stats,
            'grpo_stats': grpo_stats,
            'improvement': improvement,
            'pct_improvement': pct_improvement
        }, f, indent=2)
    
    print(f"\n✅ Saved detailed results to {output_file}")
    
    return {
        'base_stats': base_stats,
        'grpo_stats': grpo_stats,
        'improvement': improvement
    }


def main():
    """Main evaluation function."""
    evaluate_grpo(
        baseline_file='hw3_grpo/baseline_results.json',
        base_model_path='out-shakespeare-char/ckpt.pt',
        grpo_model_path='hw3_grpo/grpo_model_out/final_model.pt',
        data_path='data/shakespeare_char',
        num_test_prompts=50,
        num_samples_per_prompt=5
    )


if __name__ == "__main__":
    main()

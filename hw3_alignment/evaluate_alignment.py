"""
Evaluate DPO Alignment (HW3 Problem 1.4 - Final Evaluation)

This script compares the base model with the DPO-aligned model to demonstrate
that alignment improves output quality according to the reward model.

Evaluation includes:
1. Generate samples from both base and aligned models
2. Score samples using the reward model
3. Compare reward distributions
4. Show qualitative examples
"""

import os
import sys
import torch
import numpy as np
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from reward_model import RewardModel
from preference_heuristic import score_text


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a GPT model from checkpoint."""
    print(f"Loading model from {model_path}...")
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
    
    print(f"✅ Loaded model ({model.get_num_params()/1e6:.2f}M params)")
    return model, config


def load_reward_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the trained reward model."""
    print(f"Loading reward model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = RewardModel(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded reward model (val_acc={checkpoint.get('val_acc', 'N/A')})")
    return model


def load_encode_decode(data_path):
    """Load encoding/decoding functions."""
    meta_path = os.path.join(data_path, 'meta.pkl')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        itos = meta['itos']
    else:
        chars = sorted(list(set(open(os.path.join(data_path, 'input.txt'), 'r').read())))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode


@torch.no_grad()
def generate_and_evaluate(
    model,
    reward_model,
    encode,
    decode,
    prompts,
    num_samples_per_prompt=5,
    max_new_tokens=50,
    temperature=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate text samples and evaluate with reward model.
    
    Returns:
        List of dictionaries with generation results and scores
    """
    results = []
    
    for prompt in tqdm(prompts, desc="Generating"):
        # Encode prompt
        prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        for _ in range(num_samples_per_prompt):
            # Generate completion
            generated = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=200
            )
            
            # Decode
            full_text = decode(generated[0].tolist())
            completion = full_text[len(prompt):]
            
            # Truncate to block_size for reward model
            full_text_truncated = full_text[:reward_model.config.block_size]
            full_ids = torch.tensor(encode(full_text_truncated), dtype=torch.long, device=device)
            
            # Pad to block_size
            if full_ids.size(0) < reward_model.config.block_size:
                padding = torch.zeros(reward_model.config.block_size - full_ids.size(0), dtype=torch.long, device=device)
                full_ids = torch.cat([full_ids, padding])
            else:
                full_ids = full_ids[:reward_model.config.block_size]
            
            # Get reward from model
            reward = reward_model(full_ids.unsqueeze(0)).item()
            
            # Get true score (s count)
            true_score = score_text(completion)
            
            results.append({
                'prompt': prompt,
                'completion': completion,
                'full_text': full_text,
                'reward': reward,
                'true_score': true_score
            })
    
    return results


def evaluate_alignment(
    base_model_path='out-shakespeare-char/ckpt.pt',
    aligned_model_path='hw3_alignment/dpo_model_out/best_model.pt',
    reward_model_path='hw3_alignment/reward_model_out/best_model.pt',
    data_path='data/shakespeare_char',
    num_prompts=20,
    num_samples_per_prompt=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate and compare base vs aligned models.
    """
    print("="*80)
    print("EVALUATING DPO ALIGNMENT (HW3 Problem 1.4)")
    print("="*80)
    
    # Load models
    base_model, _ = load_model(base_model_path, device)
    aligned_model, _ = load_model(aligned_model_path, device)
    reward_model = load_reward_model(reward_model_path, device)
    encode, decode = load_encode_decode(data_path)
    
    # Sample prompts
    print(f"\nSampling {num_prompts} prompts...")
    data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
    
    prompts = []
    for _ in range(num_prompts):
        start_idx = np.random.randint(0, len(data) - 20)
        prompt_tokens = data[start_idx:start_idx + 20].tolist()
        prompt = decode(prompt_tokens)
        prompts.append(prompt)
    
    # Generate and evaluate base model
    print("\n" + "="*80)
    print("EVALUATING BASE MODEL")
    print("="*80)
    base_results = generate_and_evaluate(
        base_model,
        reward_model,
        encode,
        decode,
        prompts,
        num_samples_per_prompt=num_samples_per_prompt,
        device=device
    )
    
    # Generate and evaluate aligned model
    print("\n" + "="*80)
    print("EVALUATING ALIGNED MODEL")
    print("="*80)
    aligned_results = generate_and_evaluate(
        aligned_model,
        reward_model,
        encode,
        decode,
        prompts,
        num_samples_per_prompt=num_samples_per_prompt,
        device=device
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    base_rewards = [r['reward'] for r in base_results]
    aligned_rewards = [r['reward'] for r in aligned_results]
    
    base_true_scores = [r['true_score'] for r in base_results]
    aligned_true_scores = [r['true_score'] for r in aligned_results]
    
    print(f"\nTotal samples: {len(base_results)} (base), {len(aligned_results)} (aligned)")
    print("\n--- REWARD MODEL SCORES ---")
    print(f"Base model:")
    print(f"  Mean reward: {np.mean(base_rewards):.4f} ± {np.std(base_rewards):.4f}")
    print(f"  Median reward: {np.median(base_rewards):.4f}")
    print(f"  Range: [{np.min(base_rewards):.4f}, {np.max(base_rewards):.4f}]")
    
    print(f"\nAligned model:")
    print(f"  Mean reward: {np.mean(aligned_rewards):.4f} ± {np.std(aligned_rewards):.4f}")
    print(f"  Median reward: {np.median(aligned_rewards):.4f}")
    print(f"  Range: [{np.min(aligned_rewards):.4f}, {np.max(aligned_rewards):.4f}]")
    
    improvement = np.mean(aligned_rewards) - np.mean(base_rewards)
    print(f"\n✨ IMPROVEMENT: {improvement:+.4f} ({improvement/abs(np.mean(base_rewards))*100:+.2f}%)")
    
    print("\n--- TRUE 'S' COUNT SCORES ---")
    print(f"Base model:")
    print(f"  Mean 's' count: {np.mean(base_true_scores):.2f} ± {np.std(base_true_scores):.2f}")
    print(f"  Median 's' count: {np.median(base_true_scores):.2f}")
    
    print(f"\nAligned model:")
    print(f"  Mean 's' count: {np.mean(aligned_true_scores):.2f} ± {np.std(aligned_true_scores):.2f}")
    print(f"  Median 's' count: {np.median(aligned_true_scores):.2f}")
    
    true_improvement = np.mean(aligned_true_scores) - np.mean(base_true_scores)
    print(f"\n✨ IMPROVEMENT: {true_improvement:+.2f} 's' characters")
    
    # Statistical significance (t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(aligned_rewards, base_rewards)
    print(f"\nStatistical significance (t-test):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  ✅ Improvement is statistically significant (p < 0.05)")
    else:
        print(f"  ⚠️  Improvement is not statistically significant (p >= 0.05)")
    
    # Show examples
    print("\n" + "="*80)
    print("EXAMPLE COMPARISONS")
    print("="*80)
    
    # Sort by reward
    base_sorted = sorted(base_results, key=lambda x: x['reward'], reverse=True)
    aligned_sorted = sorted(aligned_results, key=lambda x: x['reward'], reverse=True)
    
    print("\n--- TOP 3 BASE MODEL OUTPUTS ---")
    for i, result in enumerate(base_sorted[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Reward: {result['reward']:.4f}, True 's' count: {result['true_score']}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    print("\n--- TOP 3 ALIGNED MODEL OUTPUTS ---")
    for i, result in enumerate(aligned_sorted[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Reward: {result['reward']:.4f}, True 's' count: {result['true_score']}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    print("\n--- BOTTOM 3 BASE MODEL OUTPUTS ---")
    for i, result in enumerate(base_sorted[-3:]):
        print(f"\nExample {i+1}:")
        print(f"  Reward: {result['reward']:.4f}, True 's' count: {result['true_score']}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    print("\n--- BOTTOM 3 ALIGNED MODEL OUTPUTS ---")
    for i, result in enumerate(aligned_sorted[-3:]):
        print(f"\nExample {i+1}:")
        print(f"  Reward: {result['reward']:.4f}, True 's' count: {result['true_score']}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    # Reward distribution by bins
    print("\n" + "="*80)
    print("REWARD DISTRIBUTION")
    print("="*80)
    
    bins = [-3, -1, 0, 1, 3]
    bin_labels = ['Very Low', 'Low', 'Medium', 'High']
    
    print("\nBase model distribution:")
    for i in range(len(bins) - 1):
        count = sum(1 for r in base_rewards if bins[i] <= r < bins[i+1])
        pct = count / len(base_rewards) * 100
        print(f"  {bin_labels[i]:12s}: {count:3d} samples ({pct:5.1f}%)")
    
    print("\nAligned model distribution:")
    for i in range(len(bins) - 1):
        count = sum(1 for r in aligned_rewards if bins[i] <= r < bins[i+1])
        pct = count / len(aligned_rewards) * 100
        print(f"  {bin_labels[i]:12s}: {count:3d} samples ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  • DPO alignment {'IMPROVED' if improvement > 0 else 'DECREASED'} average reward by {abs(improvement):.4f}")
    print(f"  • Average 's' count {'increased' if true_improvement > 0 else 'decreased'} by {abs(true_improvement):.2f}")
    print(f"  • Aligned model produces {'higher' if improvement > 0 else 'lower'}-quality outputs according to reward model")
    
    return {
        'base_results': base_results,
        'aligned_results': aligned_results,
        'base_rewards': base_rewards,
        'aligned_rewards': aligned_rewards,
        'improvement': improvement,
        'p_value': p_value
    }


def main():
    """Main evaluation function."""
    try:
        evaluate_alignment(
            base_model_path='out-shakespeare-char/ckpt.pt',
            aligned_model_path='hw3_alignment/dpo_model_out/best_model.pt',
            reward_model_path='hw3_alignment/reward_model_out/best_model.pt',
            data_path='data/shakespeare_char',
            num_prompts=20,
            num_samples_per_prompt=5
        )
    except ImportError as e:
        if 'scipy' in str(e):
            print("\n⚠️  scipy not available, skipping statistical test")
            print("Install with: pip install scipy")
        else:
            raise


if __name__ == "__main__":
    main()

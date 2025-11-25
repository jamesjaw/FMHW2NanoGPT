"""
Test Reward Model (HW3 Problem 1.3)

This script tests the trained reward model by:
1. Loading the trained reward model
2. Generating text samples from the base model
3. Computing rewards for the samples
4. Displaying high-reward and low-reward examples
"""

import os
import sys
import torch
import numpy as np
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from reward_model import RewardModel
from preference_heuristic import score_text


def load_reward_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a trained reward model."""
    print(f"Loading reward model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model = RewardModel(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded reward model (val_acc={checkpoint.get('val_acc', 'N/A')})")
    return model, config


def load_gpt_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the base GPT model."""
    print(f"Loading GPT model from {model_path}...")
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
    
    print(f"✅ Loaded GPT model")
    return model, config


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
def generate_and_score(
    gpt_model,
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
    Generate text samples and score them with the reward model.
    
    Returns:
        List of dictionaries with 'prompt', 'completion', 'reward', 'true_score'
    """
    results = []
    
    for prompt in prompts:
        # Encode prompt
        prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        for _ in range(num_samples_per_prompt):
            # Generate completion
            generated = gpt_model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=200
            )
            
            # Decode
            full_text = decode(generated[0].tolist())
            completion = full_text[len(prompt):]
            
            # Get reward from model
            reward = reward_model(generated).item()
            
            # Get true score (s count)
            true_score = score_text(completion)
            
            results.append({
                'prompt': prompt,
                'completion': completion,
                'reward': reward,
                'true_score': true_score,
                'full_text': full_text
            })
    
    return results


def test_reward_model(
    reward_model_path='hw3_alignment/reward_model_out/best_model.pt',
    gpt_model_path='out-shakespeare-char/ckpt.pt',
    data_path='data/shakespeare_char',
    num_prompts=10,
    num_samples_per_prompt=5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Test the reward model and display results.
    """
    print("="*80)
    print("TESTING REWARD MODEL (HW3 Problem 1.3)")
    print("="*80)
    
    # Load models
    reward_model, reward_config = load_reward_model(reward_model_path, device)
    gpt_model, gpt_config = load_gpt_model(gpt_model_path, device)
    encode, decode = load_encode_decode(data_path)
    
    # Sample some prompts
    print(f"\nSampling {num_prompts} prompts...")
    data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
    
    prompts = []
    for _ in range(num_prompts):
        start_idx = np.random.randint(0, len(data) - 20)
        prompt_tokens = data[start_idx:start_idx + 20].tolist()
        prompt = decode(prompt_tokens)
        prompts.append(prompt)
    
    # Generate and score
    print(f"\nGenerating {num_samples_per_prompt} completions per prompt...")
    results = generate_and_score(
        gpt_model,
        reward_model,
        encode,
        decode,
        prompts,
        num_samples_per_prompt=num_samples_per_prompt,
        device=device
    )
    
    # Sort by reward
    results_sorted = sorted(results, key=lambda x: x['reward'], reverse=True)
    
    # Analyze correlation
    rewards = [r['reward'] for r in results]
    true_scores = [r['true_score'] for r in results]
    correlation = np.corrcoef(rewards, true_scores)[0, 1]
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal samples: {len(results)}")
    print(f"Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")
    print(f"True score range: [{min(true_scores)}, {max(true_scores)}]")
    print(f"Correlation (reward vs true_score): {correlation:.4f}")
    
    # Show high-reward examples
    print("\n" + "="*80)
    print("HIGH-REWARD TEXT (Top 5)")
    print("="*80)
    for i, result in enumerate(results_sorted[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Reward: {result['reward']:.4f}")
        print(f"True 's' count: {result['true_score']}")
        print(f"Prompt: {result['prompt'][:50]}...")
        print(f"Completion: {result['completion'][:100]}...")
        print(f"Full text: {result['full_text'][:150]}...")
    
    # Show low-reward examples
    print("\n" + "="*80)
    print("LOW-REWARD TEXT (Bottom 5)")
    print("="*80)
    for i, result in enumerate(results_sorted[-5:]):
        print(f"\n--- Example {i+1} ---")
        print(f"Reward: {result['reward']:.4f}")
        print(f"True 's' count: {result['true_score']}")
        print(f"Prompt: {result['prompt'][:50]}...")
        print(f"Completion: {result['completion'][:100]}...")
        print(f"Full text: {result['full_text'][:150]}...")
    
    # Analyze by true score bins
    print("\n" + "="*80)
    print("REWARD BY TRUE SCORE BINS")
    print("="*80)
    
    bins = [0, 2, 4, 6, 100]
    bin_labels = ['0-1', '2-3', '4-5', '6+']
    
    for i in range(len(bins) - 1):
        bin_results = [r for r in results if bins[i] <= r['true_score'] < bins[i+1]]
        if bin_results:
            avg_reward = np.mean([r['reward'] for r in bin_results])
            print(f"True score {bin_labels[i]}: {len(bin_results)} samples, avg reward = {avg_reward:.4f}")
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE!")
    print("="*80)
    
    return results


def main():
    """Main testing function."""
    test_reward_model(
        reward_model_path='hw3_alignment/reward_model_out/best_model.pt',
        gpt_model_path='out-shakespeare-char/ckpt.pt',
        data_path='data/shakespeare_char',
        num_prompts=10,
        num_samples_per_prompt=5
    )


if __name__ == "__main__":
    main()

"""
Prepare Prompts and Baseline (HW3 Problem 2.2)

This script:
1. Creates a set of prompts from the Shakespeare dataset
2. Generates completions from the base NanoGPT model
3. Computes verifier scores for baseline
4. Reports mean score and representative examples
"""

import os
import sys
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from verifier import verifier, batch_verifier, get_verifier_stats, MAX_TOKENS


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


def create_prompts(data_path, num_prompts=100, prompt_length=15, seed=1337):
    """
    Create a set of prompts from the dataset.
    
    Args:
        data_path: Path to dataset directory
        num_prompts: Number of prompts to create
        prompt_length: Length of each prompt in tokens
        seed: Random seed
        
    Returns:
        prompts: List of prompt strings
    """
    np.random.seed(seed)
    
    # Load data
    data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
    
    # Load decode function
    _, decode = load_encode_decode(data_path)
    
    prompts = []
    for _ in range(num_prompts):
        # Random starting position
        start_idx = np.random.randint(0, len(data) - prompt_length - 1)
        prompt_tokens = data[start_idx:start_idx + prompt_length].tolist()
        prompt_text = decode(prompt_tokens)
        prompts.append(prompt_text)
    
    return prompts


@torch.no_grad()
def generate_baseline_completions(
    model,
    prompts,
    encode,
    decode,
    max_new_tokens=MAX_TOKENS,
    temperature=1.0,
    top_k=200,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Generate completions from the base model for all prompts.
    
    Args:
        model: GPT model
        prompts: List of prompt strings
        encode: Encoding function
        decode: Decoding function
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run on
        
    Returns:
        results: List of dicts with prompt, completion, full_text, score
    """
    results = []
    
    for prompt in tqdm(prompts, desc="Generating baseline"):
        # Encode prompt
        prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
        
        # Decode
        full_text = decode(generated[0].tolist())
        completion = full_text[len(prompt):]
        
        # Compute verifier score
        score = verifier(completion)
        
        results.append({
            'prompt': prompt,
            'completion': completion,
            'full_text': full_text,
            'score': score
        })
    
    return results


def prepare_baseline(
    model_path='out-shakespeare-char/ckpt.pt',
    data_path='data/shakespeare_char',
    num_prompts=100,
    prompt_length=15,
    output_file='hw3_grpo/baseline_results.json',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Prepare prompts and baseline completions.
    
    Args:
        model_path: Path to base model checkpoint
        data_path: Path to dataset
        num_prompts: Number of prompts to create
        prompt_length: Length of prompts in tokens
        output_file: Path to save results
        device: Device to run on
    """
    print("="*80)
    print("PREPARING PROMPTS AND BASELINE (HW3 Problem 2.2)")
    print("="*80)
    
    # Load model
    model, config = load_model(model_path, device)
    encode, decode = load_encode_decode(data_path)
    
    # Create prompts
    print(f"\nCreating {num_prompts} prompts...")
    prompts = create_prompts(data_path, num_prompts, prompt_length)
    print(f"✅ Created {len(prompts)} prompts")
    
    # Generate baseline completions
    print(f"\nGenerating baseline completions...")
    results = generate_baseline_completions(
        model,
        prompts,
        encode,
        decode,
        device=device
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("BASELINE RESULTS")
    print("="*80)
    
    scores = [r['score'] for r in results]
    completions = [r['completion'] for r in results]
    
    stats = get_verifier_stats(completions)
    
    print(f"\nTotal samples: {len(results)}")
    print(f"\nVerifier Score Statistics:")
    for key, value in stats.items():
        if key != 'count':
            print(f"  {key.capitalize()}: {value:.2f}")
    
    # Show representative examples
    print("\n" + "="*80)
    print("REPRESENTATIVE EXAMPLES")
    print("="*80)
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    # High score examples
    print("\n--- HIGH SCORE EXAMPLES (Top 3) ---")
    for i, result in enumerate(sorted_results[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    # Medium score examples
    mid_idx = len(sorted_results) // 2
    print("\n--- MEDIUM SCORE EXAMPLES (Middle 3) ---")
    for i, result in enumerate(sorted_results[mid_idx:mid_idx+3]):
        print(f"\nExample {i+1}:")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    # Low score examples
    print("\n--- LOW SCORE EXAMPLES (Bottom 3) ---")
    for i, result in enumerate(sorted_results[-3:]):
        print(f"\nExample {i+1}:")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Prompt: {result['prompt'][:40]}...")
        print(f"  Completion: {result['completion'][:80]}...")
    
    # Score distribution
    print("\n" + "="*80)
    print("SCORE DISTRIBUTION")
    print("="*80)
    
    bins = [0, 5, 8, 11, 15]
    bin_labels = ['Very Low (0-5)', 'Low (5-8)', 'Medium (8-11)', 'High (11-15)']
    
    for i in range(len(bins) - 1):
        count = sum(1 for s in scores if bins[i] <= s < bins[i+1])
        pct = count / len(scores) * 100
        print(f"  {bin_labels[i]:20s}: {count:3d} samples ({pct:5.1f}%)")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_data = {
        'prompts': prompts,
        'results': results,
        'statistics': stats,
        'metadata': {
            'num_prompts': num_prompts,
            'prompt_length': prompt_length,
            'max_new_tokens': MAX_TOKENS,
            'model_path': model_path,
            'verifier': 'e_count + 0.1*length'
        }
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved results to {output_file}")
    
    print("\n" + "="*80)
    print("✅ BASELINE PREPARATION COMPLETE!")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"  • {len(prompts)} prompts created")
    print(f"  • Mean baseline score: {stats['mean']:.2f}")
    print(f"  • Score range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  • Ready for GRPO training")
    
    return results, prompts


def main():
    """Main function."""
    prepare_baseline(
        model_path='out-shakespeare-char/ckpt.pt',
        data_path='data/shakespeare_char',
        num_prompts=100,
        prompt_length=15,
        output_file='hw3_grpo/baseline_results.json'
    )


if __name__ == "__main__":
    main()

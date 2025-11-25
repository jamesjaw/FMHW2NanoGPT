"""
Generate Preference Dataset for DPO Training (HW3 Problem 1.1)

This script:
1. Loads the pretrained shakespeare-char model
2. Samples prompts from the shakespeare dataset
3. Generates multiple completions for each prompt
4. Creates preference pairs using the 's' count heuristic
5. Saves the dataset for reward model and DPO training
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from preference_heuristic import score_text, create_preference_pair, analyze_text_distribution


class PreferenceDataGenerator:
    """Generate preference pairs from a pretrained model."""
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        seed: int = 1337
    ):
        """
        Initialize the preference data generator.
        
        Args:
            model_path: Path to the checkpoint (e.g., 'out-shakespeare-char/ckpt.pt')
            data_path: Path to the dataset directory (e.g., 'data/shakespeare_char')
            device: Device to run on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
        """
        self.device = device
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"Loading model from {model_path}...")
        self.model, self.config = self._load_model(model_path)
        
        print(f"Loading dataset from {data_path}...")
        self.data, self.meta = self._load_data(data_path)
        
        # Decode/encode functions
        if self.meta and 'stoi' in self.meta and 'itos' in self.meta:
            self.stoi = self.meta['stoi']
            self.itos = self.meta['itos']
            self.encode = lambda s: [self.stoi[c] for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
        else:
            print("Warning: No meta.pkl found, using default encoding")
            # Fallback to simple character encoding
            chars = sorted(list(set(open(data_path + '/input.txt', 'r').read())))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}
            self.encode = lambda s: [self.stoi[c] for c in s]
            self.decode = lambda l: ''.join([self.itos[i] for i in l])
    
    def _load_model(self, model_path: str):
        """Load the pretrained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config
        checkpoint_model_args = checkpoint['model_args']
        gptconf = GPTConfig(**checkpoint_model_args)
        model = GPT(gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        # Fix the keys if needed (remove 'module.' prefix from DDP)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        
        return model, gptconf
    
    def _load_data(self, data_path: str):
        """Load the dataset and metadata."""
        # Load metadata
        meta_path = os.path.join(data_path, 'meta.pkl')
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
        
        # Load train data
        train_data = np.memmap(os.path.join(data_path, 'train.bin'), dtype=np.uint16, mode='r')
        
        return train_data, meta
    
    def sample_prompts(self, num_prompts: int, prompt_length: int = 20) -> list:
        """
        Sample random prompts from the dataset.
        
        Args:
            num_prompts: Number of prompts to sample
            prompt_length: Length of each prompt in tokens
            
        Returns:
            List of prompt strings
        """
        prompts = []
        data_len = len(self.data)
        
        for _ in range(num_prompts):
            # Random starting position
            start_idx = np.random.randint(0, data_len - prompt_length - 1)
            prompt_tokens = self.data[start_idx:start_idx + prompt_length].tolist()
            prompt_text = self.decode(prompt_tokens)
            prompts.append(prompt_text)
        
        return prompts
    
    @torch.no_grad()
    def generate_completions(
        self,
        prompt: str,
        num_completions: int = 6,
        max_new_tokens: int = 50,
        temperatures: list = [0.8, 1.0, 1.2],
        top_k: int = 200
    ) -> list:
        """
        Generate multiple completions for a given prompt.
        
        Args:
            prompt: Input prompt string
            num_completions: Number of completions to generate
            max_new_tokens: Maximum number of new tokens to generate
            temperatures: List of temperatures to sample from
            top_k: Top-k sampling parameter
            
        Returns:
            List of completion strings (without the prompt)
        """
        completions = []
        
        # Encode prompt
        prompt_tokens = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device)
        prompt_tokens = prompt_tokens.unsqueeze(0)  # Add batch dimension
        
        for i in range(num_completions):
            # Cycle through temperatures
            temp = temperatures[i % len(temperatures)]
            
            # Generate
            generated = self.model.generate(
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k
            )
            
            # Decode and extract only the completion (remove prompt)
            full_text = self.decode(generated[0].tolist())
            completion = full_text[len(prompt):]
            completions.append(completion)
        
        return completions
    
    def generate_preference_dataset(
        self,
        num_pairs: int = 300,
        prompt_length: int = 20,
        num_completions_per_prompt: int = 6,
        max_new_tokens: int = 50,
        output_file: str = 'hw3_alignment/preference_data.json'
    ):
        """
        Generate a complete preference dataset.
        
        Args:
            num_pairs: Target number of preference pairs to generate
            prompt_length: Length of prompts in tokens
            num_completions_per_prompt: Number of completions per prompt
            max_new_tokens: Max tokens to generate per completion
            output_file: Path to save the dataset
        """
        print(f"\nGenerating {num_pairs} preference pairs...")
        print(f"  Prompt length: {prompt_length} tokens")
        print(f"  Completions per prompt: {num_completions_per_prompt}")
        print(f"  Max new tokens: {max_new_tokens}")
        print()
        
        preference_pairs = []
        all_scores = []
        
        # We might need more prompts than pairs since some won't have variation
        num_prompts = int(num_pairs * 1.2)  # 20% buffer
        prompts = self.sample_prompts(num_prompts, prompt_length)
        
        for prompt in tqdm(prompts, desc="Generating preferences"):
            # Generate multiple completions
            completions = self.generate_completions(
                prompt,
                num_completions=num_completions_per_prompt,
                max_new_tokens=max_new_tokens
            )
            
            # Create preference pair
            pair = create_preference_pair(prompt, completions, return_scores=True)
            
            # Skip if no variation (all same score)
            if pair is None:
                continue
            
            preference_pairs.append(pair)
            all_scores.extend([pair['chosen_score'], pair['rejected_score']])
            
            # Stop if we have enough pairs
            if len(preference_pairs) >= num_pairs:
                break
        
        # Analyze the dataset
        print(f"\nGenerated {len(preference_pairs)} preference pairs")
        print("\nScore distribution:")
        stats = {
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'min': int(np.min(all_scores)),
            'max': int(np.max(all_scores)),
            'median': float(np.median(all_scores))
        }
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        
        # Show some examples
        print("\nExample preference pairs:")
        for i in range(min(3, len(preference_pairs))):
            pair = preference_pairs[i]
            print(f"\n--- Pair {i+1} ---")
            print(f"Prompt: {pair['prompt'][:50]}...")
            print(f"Chosen (score={pair['chosen_score']}): {pair['chosen'][:50]}...")
            print(f"Rejected (score={pair['rejected_score']}): {pair['rejected'][:50]}...")
        
        # Save dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump({
                'preference_pairs': preference_pairs,
                'metadata': {
                    'num_pairs': len(preference_pairs),
                    'prompt_length': prompt_length,
                    'num_completions_per_prompt': num_completions_per_prompt,
                    'max_new_tokens': max_new_tokens,
                    'model_path': 'out-shakespeare-char/ckpt.pt',
                    'heuristic': 's_count',
                    'statistics': stats
                }
            }, f, indent=2)
        
        print(f"\nDataset saved to {output_file}")
        return preference_pairs


def main():
    """Main function to generate preference dataset."""
    
    # Configuration
    MODEL_PATH = 'out-shakespeare-char/ckpt.pt'
    DATA_PATH = 'data/shakespeare_char'
    OUTPUT_FILE = 'hw3_alignment/preference_data.json'
    
    NUM_PAIRS = 300
    PROMPT_LENGTH = 20
    NUM_COMPLETIONS = 6
    MAX_NEW_TOKENS = 50
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train a model first or update MODEL_PATH")
        return
    
    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data not found at {DATA_PATH}")
        print("Please prepare the dataset first or update DATA_PATH")
        return
    
    # Generate dataset
    generator = PreferenceDataGenerator(
        model_path=MODEL_PATH,
        data_path=DATA_PATH
    )
    
    generator.generate_preference_dataset(
        num_pairs=NUM_PAIRS,
        prompt_length=PROMPT_LENGTH,
        num_completions_per_prompt=NUM_COMPLETIONS,
        max_new_tokens=MAX_NEW_TOKENS,
        output_file=OUTPUT_FILE
    )
    
    print("\n✅ Preference dataset generation complete!")


if __name__ == "__main__":
    main()

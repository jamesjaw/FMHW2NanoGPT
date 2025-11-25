"""
GRPO Training (Group Relative Policy Optimization) - HW3 Problem 2.3

This script implements GRPO (RLVR) to optimize the model using verifier rewards.

GRPO Algorithm:
1. Sample K completions for each prompt from current policy
2. Compute verifier scores for all completions
3. Compute advantages using group statistics (mean, std)
4. Update policy using policy gradient with advantages
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

# Optional matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from verifier import verifier, MAX_TOKENS


class GRPOTrainer:
    """
    Trainer for Group Relative Policy Optimization (GRPO).
    
    GRPO uses group-relative advantages to update the policy:
    - Sample K completions per prompt
    - Compute advantages relative to group mean
    - Update policy to increase probability of high-advantage completions
    """
    
    def __init__(
        self,
        policy_model: GPT,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            policy_model: The model to train
            learning_rate: Learning rate
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient (not used in pure GRPO)
            entropy_coef: Entropy bonus coefficient
            device: Device to train on
        """
        self.policy_model = policy_model.to(device)
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.block_size = policy_model.config.block_size  # Store block_size
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=learning_rate,
            weight_decay=0.0
        )
        
        print(f"GRPO Trainer initialized:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Clip epsilon: {clip_epsilon}")
        print(f"  Entropy coef: {entropy_coef}")
        print(f"  Block size: {self.block_size}")
    
    def compute_log_probs(self, input_ids):
        """
        Compute log probabilities for a sequence.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            
        Returns:
            log_probs: Log probability of each token (batch_size, seq_len-1)
            entropy: Entropy of the distribution (batch_size, seq_len-1)
        """
        batch_size, seq_len = input_ids.shape
        
        if seq_len < 2:
            return torch.zeros(batch_size, 1, device=self.device), torch.zeros(batch_size, 1, device=self.device)
        
        # Forward pass
        logits, _ = self.policy_model(input_ids, targets=input_ids)
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Log probabilities
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        log_probs = torch.gather(
            log_probs_all,
            dim=2,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Entropy
        probs = F.softmax(shift_logits, dim=-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        
        return log_probs, entropy
    
    def compute_grpo_loss(self, sequences, rewards, old_log_probs=None):
        """
        Compute GRPO loss.
        
        Args:
            sequences: List of token ID tensors (varying lengths)
            rewards: Verifier rewards for each sequence (batch_size,)
            old_log_probs: Old log probs for PPO clipping (optional)
            
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        # Truncate sequences to block_size and pad to same length
        truncated_seqs = []
        for seq in sequences:
            if seq.size(0) > self.block_size:
                truncated_seqs.append(seq[:self.block_size])
            else:
                truncated_seqs.append(seq)
        
        max_len = min(max(seq.size(0) for seq in truncated_seqs), self.block_size)
        padded_seqs = []
        masks = []
        
        for seq in truncated_seqs:
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                padded = F.pad(seq, (0, pad_len), value=0)
            else:
                padded = seq
            padded_seqs.append(padded)
            
            # Create mask (1 for real tokens, 0 for padding)
            mask = torch.ones(seq.size(0), device=self.device)
            if pad_len > 0:
                mask = F.pad(mask, (0, pad_len), value=0)
            masks.append(mask)
        
        # Stack
        input_ids = torch.stack(padded_seqs)  # (batch_size, max_len)
        masks = torch.stack(masks)  # (batch_size, max_len)
        
        # Compute log probs and entropy
        log_probs, entropy = self.compute_log_probs(input_ids)  # (batch_size, max_len-1)
        
        # Mask out padding
        mask_shifted = masks[:, 1:]  # Shift mask to match log_probs
        
        # Sum log probs over sequence (masked)
        sequence_log_probs = (log_probs * mask_shifted).sum(dim=1) / (mask_shifted.sum(dim=1) + 1e-8)
        
        # Compute advantages using group statistics
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        # Policy gradient loss
        pg_loss = -(sequence_log_probs * advantages).mean()
        
        # Entropy bonus (encourage exploration)
        entropy_bonus = (entropy * mask_shifted).sum(dim=1) / (mask_shifted.sum(dim=1) + 1e-8)
        entropy_loss = -self.entropy_coef * entropy_bonus.mean()
        
        # Total loss
        total_loss = pg_loss + entropy_loss
        
        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_reward': mean_reward.item(),
            'std_reward': std_reward.item(),
            'mean_advantage': advantages.mean().item(),
        }
        
        return total_loss, metrics
    
    def train_step(self, sequences, rewards):
        """
        Single training step.
        
        Args:
            sequences: List of token ID tensors
            rewards: Verifier rewards
            
        Returns:
            metrics: Training metrics
        """
        self.policy_model.train()
        
        # Compute loss
        loss, metrics = self.compute_grpo_loss(sequences, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        
        return metrics


def load_baseline_data(baseline_file):
    """Load baseline data."""
    with open(baseline_file, 'r') as f:
        data = json.load(f)
    return data['prompts'], data['results']


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
    return model, config


def load_encode_decode(data_path):
    """Load encode/decode functions."""
    import pickle
    meta_path = os.path.join(data_path, 'meta.pkl')
    
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    itos = meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode


@torch.no_grad()
def sample_completions(
    model,
    prompt,
    encode,
    decode,
    num_samples=4,
    max_new_tokens=MAX_TOKENS,
    temperature=1.0,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Sample multiple completions for a prompt.
    
    Returns:
        completions: List of completion strings
        sequences: List of full sequence tensors (prompt + completion)
        rewards: List of verifier scores
    """
    prompt_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    completions = []
    sequences = []
    rewards = []
    
    for _ in range(num_samples):
        # Generate
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=200
        )
        
        # Decode
        full_text = decode(generated[0].tolist())
        completion = full_text[len(prompt):]
        
        # Compute reward
        reward = verifier(completion)
        
        completions.append(completion)
        sequences.append(generated[0])
        rewards.append(reward)
    
    return completions, sequences, rewards


def train_grpo(
    baseline_file='hw3_grpo/baseline_results.json',
    model_path='out-shakespeare-char/ckpt.pt',
    data_path='data/shakespeare_char',
    output_dir='hw3_grpo/grpo_model_out',
    # GRPO config
    learning_rate=1e-5,
    num_epochs=3,
    samples_per_prompt=4,
    batch_size=8,  # Number of prompts per batch
    # Logging
    eval_interval=10,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train model using GRPO.
    """
    print("="*80)
    print("GRPO TRAINING (HW3 Problem 2.3)")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading baseline data...")
    prompts, baseline_results = load_baseline_data(baseline_file)
    print(f"✅ Loaded {len(prompts)} prompts")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model, config = load_model(model_path, device)
    model.to(device)
    print(f"✅ Loaded model ({model.get_num_params()/1e6:.2f}M params)")
    
    # Load encode/decode
    encode, decode = load_encode_decode(data_path)
    
    # Create trainer
    trainer = GRPOTrainer(
        policy_model=model,
        learning_rate=learning_rate,
        device=device
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING GRPO TRAINING")
    print("="*80)
    print(f"Epochs: {num_epochs}")
    print(f"Samples per prompt: {samples_per_prompt}")
    print(f"Batch size: {batch_size}")
    
    all_rewards = []
    steps = []
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Shuffle prompts
        np.random.shuffle(prompts)
        
        epoch_rewards = []
        
        # Process in batches
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc=f"Epoch {epoch+1}"):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            
            batch_sequences = []
            batch_rewards = []
            
            # Sample completions for each prompt in batch
            for prompt in batch_prompts:
                completions, sequences, rewards = sample_completions(
                    model,
                    prompt,
                    encode,
                    decode,
                    num_samples=samples_per_prompt,
                    device=device
                )
                
                batch_sequences.extend(sequences)
                batch_rewards.extend(rewards)
                epoch_rewards.extend(rewards)
            
            # Train step
            metrics = trainer.train_step(batch_sequences, batch_rewards)
            
            # Logging
            if global_step % eval_interval == 0:
                mean_reward = np.mean(batch_rewards)
                all_rewards.append(mean_reward)
                steps.append(global_step)
                
                print(f"\nStep {global_step}:")
                print(f"  Mean reward: {mean_reward:.2f}")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  PG loss: {metrics['pg_loss']:.4f}")
            
            global_step += 1
        
        # Epoch summary
        epoch_mean = np.mean(epoch_rewards)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Mean reward: {epoch_mean:.2f}")
        print(f"  Std reward: {np.std(epoch_rewards):.2f}")
        
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'model_args': config.__dict__,
            'epoch': epoch,
            'step': global_step,
        }
        torch.save(checkpoint, os.path.join(output_dir, f'epoch_{epoch+1}.pt'))
    
    # Save final model
    final_checkpoint = {
        'model': model.state_dict(),
        'model_args': config.__dict__,
        'step': global_step,
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final_model.pt'))
    print(f"\n✅ Saved final model to {output_dir}/final_model.pt")
    
    # Plot training curve
    if HAS_MATPLOTLIB and len(all_rewards) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, all_rewards, marker='o')
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Verifier Reward')
        plt.title('GRPO Training: Reward Evolution')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=150)
        print(f"✅ Saved training curve to {output_dir}/training_curve.png")
    
    print("\n" + "="*80)
    print("✅ GRPO TRAINING COMPLETE!")
    print("="*80)
    
    return model


def main():
    """Main training function."""
    train_grpo(
        baseline_file='hw3_grpo/baseline_results.json',
        model_path='out-shakespeare-char/ckpt.pt',
        data_path='data/shakespeare_char',
        output_dir='hw3_grpo/grpo_model_out',
        learning_rate=1e-5,
        num_epochs=3,
        samples_per_prompt=4,
        batch_size=8,
        eval_interval=5
    )


if __name__ == "__main__":
    main()

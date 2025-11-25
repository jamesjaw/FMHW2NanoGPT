"""
DPO Training (Direct Preference Optimization) - HW3 Problem 1.4

This script implements DPO to align the NanoGPT model with the 's' count preference.
DPO directly optimizes the policy without needing explicit RL or a separate reward model.

Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/abs/2305.18290
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization (DPO).
    
    DPO optimizes the policy model directly using preference pairs,
    without needing a separate reward model or RL.
    """
    
    def __init__(
        self,
        policy_model: GPT,
        ref_model: GPT,
        beta: float = 0.1,
        learning_rate: float = 5e-6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize DPO trainer.
        
        Args:
            policy_model: The model to train (will be updated)
            ref_model: Reference model (frozen, used for KL regularization)
            beta: Temperature parameter for DPO loss
            learning_rate: Learning rate for optimizer
            device: Device to train on
        """
        self.policy_model = policy_model.to(device)
        self.ref_model = ref_model.to(device)
        self.beta = beta
        self.device = device
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # Optimizer for policy model
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=learning_rate,
            weight_decay=0.0
        )
        
        print(f"DPO Trainer initialized with beta={beta}, lr={learning_rate}")
    
    def get_log_probs(self, model, input_ids, labels):
        """
        Compute log probabilities of the labels under the model.
        
        Args:
            model: GPT model
            input_ids: Input token IDs (batch_size, seq_len)
            labels: Target token IDs (batch_size, seq_len)
        
        Returns:
            log_probs: Log probabilities of the labels (batch_size,)
        """
        batch_size, seq_len = input_ids.shape
        
        # Need at least 2 tokens for next-token prediction
        if seq_len < 2:
            return torch.zeros(batch_size, device=input_ids.device)
        
        # Forward pass - pass labels as targets to get full logits
        # (otherwise model only returns last token logits for generation)
        logits, _ = model(input_ids, targets=labels)  # (batch_size, seq_len, vocab_size)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous()  # (batch_size, seq_len-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch_size, seq_len-1, vocab_size)
        
        # Gather log probs for the actual labels
        # log_probs[i, j, labels[i, j]] for each i, j
        gathered_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, seq_len-1)
        
        # Sum over sequence length (average per token)
        # Mask out padding tokens (assuming 0 is padding)
        mask = (shift_labels != 0).float()
        
        # Avoid division by zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        
        sequence_log_probs = (gathered_log_probs * mask).sum(dim=1) / mask_sum
        
        return sequence_log_probs
    
    def compute_dpo_loss(self, chosen_ids, rejected_ids):
        """
        Compute DPO loss.
        
        DPO Loss:
        L = -log(σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)) 
                    - β * (log π_θ(y_l|x) - log π_ref(y_l|x))))
        
        where:
        - π_θ is the policy model (being trained)
        - π_ref is the reference model (frozen)
        - y_w is the chosen (preferred) completion
        - y_l is the rejected completion
        - β is the temperature parameter
        
        Args:
            chosen_ids: Token IDs for chosen completions (batch_size, seq_len)
            rejected_ids: Token IDs for rejected completions (batch_size, seq_len)
        
        Returns:
            loss: DPO loss
            metrics: Dictionary with additional metrics
        """
        # Get log probs from policy model
        policy_chosen_logprobs = self.get_log_probs(self.policy_model, chosen_ids, chosen_ids)
        policy_rejected_logprobs = self.get_log_probs(self.policy_model, rejected_ids, rejected_ids)
        
        # Get log probs from reference model
        with torch.no_grad():
            ref_chosen_logprobs = self.get_log_probs(self.ref_model, chosen_ids, chosen_ids)
            ref_rejected_logprobs = self.get_log_probs(self.ref_model, rejected_ids, rejected_ids)
        
        # Compute log ratios
        chosen_log_ratios = policy_chosen_logprobs - ref_chosen_logprobs
        rejected_log_ratios = policy_rejected_logprobs - ref_rejected_logprobs
        
        # DPO loss
        logits = self.beta * (chosen_log_ratios - rejected_log_ratios)
        loss = -F.logsigmoid(logits).mean()
        
        # Compute metrics
        with torch.no_grad():
            # Implicit reward (for monitoring)
            chosen_rewards = self.beta * chosen_log_ratios
            rejected_rewards = self.beta * rejected_log_ratios
            
            # Accuracy: how often is chosen > rejected?
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            # Reward margin
            reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'reward_margin': reward_margin.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
        }
        
        return loss, metrics
    
    def train_step(self, chosen_ids, rejected_ids):
        """
        Single training step.
        
        Args:
            chosen_ids: Token IDs for chosen completions
            rejected_ids: Token IDs for rejected completions
        
        Returns:
            metrics: Dictionary with training metrics
        """
        self.policy_model.train()
        
        # Move to device
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        
        # Compute loss
        loss, metrics = self.compute_dpo_loss(chosen_ids, rejected_ids)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        
        # Update
        self.optimizer.step()
        
        return metrics
    
    @torch.no_grad()
    def eval_step(self, chosen_ids, rejected_ids):
        """
        Single evaluation step.
        
        Args:
            chosen_ids: Token IDs for chosen completions
            rejected_ids: Token IDs for rejected completions
        
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        self.policy_model.eval()
        
        # Move to device
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        
        # Compute loss
        _, metrics = self.compute_dpo_loss(chosen_ids, rejected_ids)
        
        return metrics


class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset for preference pairs."""
    
    def __init__(self, preference_pairs, encode_fn, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            preference_pairs: List of preference pair dictionaries
            encode_fn: Function to encode text to token IDs
            max_length: Maximum sequence length
        """
        self.pairs = preference_pairs
        self.encode = encode_fn
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Encode prompt + chosen/rejected
        prompt = pair['prompt']
        chosen_text = prompt + pair['chosen']
        rejected_text = prompt + pair['rejected']
        
        chosen_ids = self.encode(chosen_text)
        rejected_ids = self.encode(rejected_text)
        
        # Truncate to max_length
        chosen_ids = chosen_ids[:self.max_length]
        rejected_ids = rejected_ids[:self.max_length]
        
        # Pad to max_length (fixed length for all sequences)
        chosen_ids = chosen_ids + [0] * (self.max_length - len(chosen_ids))
        rejected_ids = rejected_ids + [0] * (self.max_length - len(rejected_ids))
        
        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function - all sequences are already same length."""
    chosen_ids = torch.stack([item['chosen_ids'] for item in batch])
    rejected_ids = torch.stack([item['rejected_ids'] for item in batch])
    
    return {
        'chosen_ids': chosen_ids,
        'rejected_ids': rejected_ids
    }


def load_preference_data(data_file):
    """Load preference dataset from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data['preference_pairs'], data['metadata']


def load_encode_function(data_path):
    """Load the encode function from the dataset metadata."""
    import pickle
    meta_path = os.path.join(data_path, 'meta.pkl')
    
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi = meta['stoi']
        encode = lambda s: [stoi[c] for c in s]
        vocab_size = len(stoi)
    else:
        # Fallback
        print("Warning: No meta.pkl found, using default encoding")
        chars = sorted(list(set(open(os.path.join(data_path, 'input.txt'), 'r').read())))
        stoi = {ch: i for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        vocab_size = len(stoi)
    
    return encode, vocab_size


def train_dpo(
    preference_file='hw3_alignment/preference_data.json',
    data_path='data/shakespeare_char',
    model_path='out-shakespeare-char/ckpt.pt',
    output_dir='hw3_alignment/dpo_model_out',
    # DPO config
    beta=0.1,
    learning_rate=5e-6,
    # Training config
    batch_size=8,
    num_epochs=5,
    eval_interval=20,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the model using DPO.
    
    Args:
        preference_file: Path to preference dataset JSON
        data_path: Path to original dataset (for encoding)
        model_path: Path to pretrained model checkpoint
        output_dir: Directory to save model checkpoints
        beta: DPO temperature parameter
        learning_rate: Learning rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        eval_interval: Steps between evaluations
        device: Device to train on
    """
    print("="*80)
    print("DPO TRAINING (HW3 Problem 1.4)")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading preference data...")
    preference_pairs, metadata = load_preference_data(preference_file)
    print(f"Loaded {len(preference_pairs)} preference pairs")
    
    # Load encoding function
    print("\nLoading encoding function...")
    encode, vocab_size = load_encode_function(data_path)
    print(f"Vocabulary size: {vocab_size}")
    
    # Load pretrained model first to get block_size
    print(f"\nLoading pretrained model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    print(f"Model config: block_size={config.block_size}, vocab_size={config.vocab_size}")
    
    # Split into train/val
    split_idx = int(0.9 * len(preference_pairs))
    train_pairs = preference_pairs[:split_idx]
    val_pairs = preference_pairs[split_idx:]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create datasets with model's block_size
    train_dataset = PreferenceDataset(train_pairs, encode, max_length=config.block_size)
    val_dataset = PreferenceDataset(val_pairs, encode, max_length=config.block_size)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create reference model (frozen) - reuse already loaded checkpoint
    ref_model = GPT(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    ref_model.load_state_dict(state_dict)
    print(f"✅ Loaded reference model ({ref_model.get_num_params()/1e6:.2f}M params)")
    
    # Create policy model (copy of reference, will be trained)
    policy_model = GPT(config)
    policy_model.load_state_dict(copy.deepcopy(state_dict))
    print(f"✅ Created policy model ({policy_model.get_num_params()/1e6:.2f}M params)")
    
    # Create DPO trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        beta=beta,
        learning_rate=learning_rate,
        device=device
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING DPO TRAINING")
    print("="*80)
    
    train_losses = []
    train_accs = []
    train_margins = []
    val_losses = []
    val_accs = []
    val_margins = []
    steps = []
    
    global_step = 0
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        epoch_metrics = {
            'loss': [],
            'accuracy': [],
            'reward_margin': []
        }
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            metrics = trainer.train_step(batch['chosen_ids'], batch['rejected_ids'])
            
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'margin': f"{metrics['reward_margin']:.4f}"
            })
            
            # Evaluation
            if global_step % eval_interval == 0:
                val_metrics = {
                    'loss': [],
                    'accuracy': [],
                    'reward_margin': []
                }
                
                for val_batch in val_loader:
                    v_metrics = trainer.eval_step(val_batch['chosen_ids'], val_batch['rejected_ids'])
                    for key in val_metrics:
                        val_metrics[key].append(v_metrics[key])
                
                avg_val_loss = np.mean(val_metrics['loss'])
                avg_val_acc = np.mean(val_metrics['accuracy'])
                avg_val_margin = np.mean(val_metrics['reward_margin'])
                
                # Record metrics
                recent_steps = min(eval_interval, len(epoch_metrics['loss']))
                train_losses.append(np.mean(epoch_metrics['loss'][-recent_steps:]))
                train_accs.append(np.mean(epoch_metrics['accuracy'][-recent_steps:]))
                train_margins.append(np.mean(epoch_metrics['reward_margin'][-recent_steps:]))
                val_losses.append(avg_val_loss)
                val_accs.append(avg_val_acc)
                val_margins.append(avg_val_margin)
                steps.append(global_step)
                
                print(f"\nStep {global_step}:")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val Margin: {avg_val_margin:.4f}")
                
                # Save best model
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    checkpoint = {
                        'model': policy_model.state_dict(),
                        'model_args': config.__dict__,
                        'step': global_step,
                        'val_acc': avg_val_acc,
                        'val_loss': avg_val_loss,
                        'dpo_beta': beta
                    }
                    torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
                    print(f"  ✅ Saved best model (acc={best_val_acc:.4f})")
            
            global_step += 1
        
        # Epoch summary
        avg_train_loss = np.mean(epoch_metrics['loss'])
        avg_train_acc = np.mean(epoch_metrics['accuracy'])
        avg_train_margin = np.mean(epoch_metrics['reward_margin'])
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}, Margin: {avg_train_margin:.4f}")
    
    # Save final model
    final_checkpoint = {
        'model': policy_model.state_dict(),
        'model_args': config.__dict__,
        'step': global_step,
        'dpo_beta': beta
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final_model.pt'))
    print(f"\n✅ Saved final model to {output_dir}/final_model.pt")
    
    # Plot training curves (if matplotlib available)
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(steps, train_losses, label='Train Loss', marker='o')
        axes[0].plot(steps, val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('DPO Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy
        axes[1].plot(steps, train_accs, label='Train Acc', marker='o')
        axes[1].plot(steps, val_accs, label='Val Acc', marker='s')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Preference Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Reward Margin
        axes[2].plot(steps, train_margins, label='Train Margin', marker='o')
        axes[2].plot(steps, val_margins, label='Val Margin', marker='s')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Reward Margin')
        axes[2].set_title('Reward Margin (Chosen - Rejected)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dpo_training_curves.png'), dpi=150)
        print(f"✅ Saved training curves to {output_dir}/dpo_training_curves.png")
    else:
        print("⚠️  Skipping plot generation (matplotlib not available)")
    
    print("\n" + "="*80)
    print("DPO TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*80)
    
    return policy_model


def main():
    """Main DPO training function."""
    train_dpo(
        preference_file='hw3_alignment/preference_data.json',
        data_path='data/shakespeare_char',
        model_path='out-shakespeare-char/ckpt.pt',
        output_dir='hw3_alignment/dpo_model_out',
        beta=0.1,
        learning_rate=5e-6,
        batch_size=8,
        num_epochs=5,
        eval_interval=20
    )


if __name__ == "__main__":
    main()

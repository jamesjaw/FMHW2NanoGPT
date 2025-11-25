"""
Train Reward Model (HW3 Problem 1.2)

This script trains a reward model on preference pairs using the Bradley-Terry model.
The reward model learns to predict which completions are preferred based on the
's' count heuristic.
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# Optional matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, plots will not be generated")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig
from reward_model import RewardModel, RewardModelTrainer


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
        
        # Truncate if needed
        chosen_ids = chosen_ids[:self.max_length]
        rejected_ids = rejected_ids[:self.max_length]
        
        # Pad to same length (for batching)
        max_len = max(len(chosen_ids), len(rejected_ids))
        chosen_ids = chosen_ids + [0] * (max_len - len(chosen_ids))
        rejected_ids = rejected_ids + [0] * (max_len - len(rejected_ids))
        
        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Find max length in batch
    max_len = max(
        max(item['chosen_ids'].size(0), item['rejected_ids'].size(0))
        for item in batch
    )
    
    # Pad all sequences to max length
    chosen_ids = []
    rejected_ids = []
    
    for item in batch:
        c = item['chosen_ids']
        r = item['rejected_ids']
        
        # Pad
        c_padded = torch.cat([c, torch.zeros(max_len - c.size(0), dtype=torch.long)])
        r_padded = torch.cat([r, torch.zeros(max_len - r.size(0), dtype=torch.long)])
        
        chosen_ids.append(c_padded)
        rejected_ids.append(r_padded)
    
    return {
        'chosen_ids': torch.stack(chosen_ids),
        'rejected_ids': torch.stack(rejected_ids)
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


def train_reward_model(
    preference_file='hw3_alignment/preference_data.json',
    data_path='data/shakespeare_char',
    output_dir='hw3_alignment/reward_model_out',
    pretrained_model_path='out-shakespeare-char/ckpt.pt',
    # Model config
    n_layer=4,
    n_head=4,
    n_embd=128,
    # Training config
    batch_size=16,
    learning_rate=1e-4,
    num_epochs=10,
    eval_interval=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the reward model.
    
    Args:
        preference_file: Path to preference dataset JSON
        data_path: Path to original dataset (for encoding)
        output_dir: Directory to save model checkpoints
        pretrained_model_path: Optional path to pretrained GPT model
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        eval_interval: Steps between evaluations
        device: Device to train on
    """
    print("="*60)
    print("TRAINING REWARD MODEL (HW3 Problem 1.2)")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading preference data...")
    preference_pairs, metadata = load_preference_data(preference_file)
    print(f"Loaded {len(preference_pairs)} preference pairs")
    print(f"Metadata: {metadata}")
    
    # Load encoding function
    print("\nLoading encoding function...")
    encode, vocab_size = load_encode_function(data_path)
    print(f"Vocabulary size: {vocab_size}")
    
    # Split into train/val
    split_idx = int(0.9 * len(preference_pairs))
    train_pairs = preference_pairs[:split_idx]
    val_pairs = preference_pairs[split_idx:]
    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Create datasets
    train_dataset = PreferenceDataset(train_pairs, encode)
    val_dataset = PreferenceDataset(val_pairs, encode)
    
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
    
    # Create model
    print("\nCreating reward model...")
    config = GPTConfig(
        block_size=128,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.1,
        bias=False
    )
    
    reward_model = RewardModel(config, pooling='last')
    
    # Optionally load pretrained backbone
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"\nLoading pretrained backbone from {pretrained_model_path}...")
        checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)
        gpt_config = GPTConfig(**checkpoint['model_args'])
        gpt_model = GPT(gpt_config)
        
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        gpt_model.load_state_dict(state_dict)
        
        # Only load if ALL dimensions match
        if (gpt_config.n_embd == config.n_embd and 
            gpt_config.vocab_size == config.vocab_size and
            gpt_config.block_size == config.block_size):
            reward_model.load_pretrained_backbone(gpt_model)
        else:
            print(f"Warning: Pretrained model dimensions don't match:")
            print(f"  GPT: n_embd={gpt_config.n_embd}, vocab={gpt_config.vocab_size}, block={gpt_config.block_size}")
            print(f"  Reward: n_embd={config.n_embd}, vocab={config.vocab_size}, block={config.block_size}")
            print("  Training from scratch instead")
    else:
        print("\nNo pretrained model found, training from scratch")
    
    # Create trainer
    trainer = RewardModelTrainer(
        reward_model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    steps = []
    
    global_step = 0
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        epoch_losses = []
        epoch_accs = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            loss, acc = trainer.train_step(batch['chosen_ids'], batch['rejected_ids'])
            epoch_losses.append(loss)
            epoch_accs.append(acc)
            
            pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}'})
            
            # Evaluation
            if global_step % eval_interval == 0:
                val_loss_list = []
                val_acc_list = []
                
                for val_batch in val_loader:
                    v_loss, v_acc = trainer.eval_step(val_batch['chosen_ids'], val_batch['rejected_ids'])
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                
                avg_val_loss = np.mean(val_loss_list)
                avg_val_acc = np.mean(val_acc_list)
                
                train_losses.append(np.mean(epoch_losses[-eval_interval:] if len(epoch_losses) >= eval_interval else epoch_losses))
                train_accs.append(np.mean(epoch_accs[-eval_interval:] if len(epoch_accs) >= eval_interval else epoch_accs))
                val_losses.append(avg_val_loss)
                val_accs.append(avg_val_acc)
                steps.append(global_step)
                
                print(f"\nStep {global_step}: Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
                
                # Save best model
                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    checkpoint = {
                        'model': reward_model.state_dict(),
                        'config': config,
                        'step': global_step,
                        'val_acc': avg_val_acc,
                        'val_loss': avg_val_loss
                    }
                    torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
                    print(f"✅ Saved best model (acc={best_val_acc:.4f})")
            
            global_step += 1
        
        # Epoch summary
        avg_train_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accs)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
    
    # Save final model
    final_checkpoint = {
        'model': reward_model.state_dict(),
        'config': config,
        'step': global_step,
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final_model.pt'))
    print(f"\n✅ Saved final model to {output_dir}/final_model.pt")
    
    # Plot training curves (if matplotlib available)
    if HAS_MATPLOTLIB:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, train_losses, label='Train Loss')
        plt.plot(steps, val_losses, label='Val Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, train_accs, label='Train Acc')
        plt.plot(steps, val_accs, label='Val Acc')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
        print(f"✅ Saved training curves to {output_dir}/training_curves.png")
    else:
        print("⚠️  Skipping plot generation (matplotlib not available)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)
    
    return reward_model, train_losses, val_losses, train_accs, val_accs


def main():
    """Main training function."""
    train_reward_model(
        preference_file='hw3_alignment/preference_data.json',
        data_path='data/shakespeare_char',
        output_dir='hw3_alignment/reward_model_out',
        pretrained_model_path='out-shakespeare-char/ckpt.pt',
        n_layer=4,
        n_head=4,
        n_embd=128,
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=10,
        eval_interval=50
    )


if __name__ == "__main__":
    main()

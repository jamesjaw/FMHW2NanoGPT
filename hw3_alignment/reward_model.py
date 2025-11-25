"""
Reward Model for NanoGPT Alignment (HW3 Problem 1.2)

This module defines a reward model that predicts scalar rewards for text.
The model is based on the GPT architecture but outputs a single scalar value
instead of next-token predictions.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Add parent directory to import GPT components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GPT, GPTConfig, LayerNorm, Block


class RewardModel(nn.Module):
    """
    Reward Model based on GPT architecture.
    
    Takes a sequence of tokens and outputs a single scalar reward.
    Uses the GPT backbone with a reward head that pools the sequence
    and produces a scalar output.
    """
    
    def __init__(self, config: GPTConfig, pooling: str = 'last'):
        """
        Initialize the reward model.
        
        Args:
            config: GPTConfig object with model hyperparameters
            pooling: How to pool the sequence ('last', 'mean', or 'max')
                - 'last': Use the last token's representation
                - 'mean': Average all token representations
                - 'max': Max pool all token representations
        """
        super().__init__()
        self.config = config
        self.pooling = pooling
        
        # GPT backbone (same as language model but without lm_head)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Reward head: maps from embedding dimension to scalar
        self.reward_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print("Reward model initialized with %.2fM parameters" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, attention_mask=None):
        """
        Forward pass of the reward model.
        
        Args:
            idx: Token indices of shape (batch_size, sequence_length)
            attention_mask: Optional mask of shape (batch_size, sequence_length)
                           1 for real tokens, 0 for padding
        
        Returns:
            rewards: Scalar rewards of shape (batch_size,)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        
        # Token and position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # (b, t, n_embd)
        
        # Pool the sequence to get a single vector per example
        if self.pooling == 'last':
            # Use the last token's representation
            if attention_mask is not None:
                # Get the last non-padding token for each sequence
                lengths = attention_mask.sum(dim=1) - 1  # (b,)
                pooled = x[torch.arange(b, device=device), lengths]  # (b, n_embd)
            else:
                pooled = x[:, -1, :]  # (b, n_embd)
        
        elif self.pooling == 'mean':
            # Average pooling
            if attention_mask is not None:
                # Masked average
                mask = attention_mask.unsqueeze(-1)  # (b, t, 1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1)  # (b, n_embd)
            else:
                pooled = x.mean(dim=1)  # (b, n_embd)
        
        elif self.pooling == 'max':
            # Max pooling
            if attention_mask is not None:
                # Masked max
                mask = attention_mask.unsqueeze(-1)  # (b, t, 1)
                x_masked = x.masked_fill(mask == 0, float('-inf'))
                pooled = x_masked.max(dim=1)[0]  # (b, n_embd)
            else:
                pooled = x.max(dim=1)[0]  # (b, n_embd)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Reward head
        rewards = self.reward_head(pooled).squeeze(-1)  # (b,)
        
        return rewards
    
    def load_pretrained_backbone(self, gpt_model: GPT):
        """
        Load pretrained weights from a GPT model into the backbone.
        This allows us to initialize the reward model with a pretrained LM.
        
        Args:
            gpt_model: Pretrained GPT model
        """
        # Copy transformer weights (excluding lm_head)
        self.transformer.wte.load_state_dict(gpt_model.transformer.wte.state_dict())
        self.transformer.wpe.load_state_dict(gpt_model.transformer.wpe.state_dict())
        self.transformer.ln_f.load_state_dict(gpt_model.transformer.ln_f.state_dict())
        
        for i, block in enumerate(self.transformer.h):
            block.load_state_dict(gpt_model.transformer.h[i].state_dict())
        
        print("✅ Loaded pretrained backbone from GPT model")


class RewardModelTrainer:
    """
    Trainer for the reward model using preference pairs.
    Implements the Bradley-Terry preference model.
    """
    
    def __init__(
        self,
        model: RewardModel,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        """
        Initialize the trainer.
        
        Args:
            model: RewardModel to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def compute_loss(self, chosen_rewards, rejected_rewards):
        """
        Compute the preference loss using the Bradley-Terry model.
        
        Loss = -log(sigmoid(r_chosen - r_rejected))
        
        This encourages the model to assign higher rewards to chosen completions.
        
        Args:
            chosen_rewards: Rewards for chosen completions (batch_size,)
            rejected_rewards: Rewards for rejected completions (batch_size,)
        
        Returns:
            loss: Scalar loss value
            accuracy: Fraction of examples where r_chosen > r_rejected
        """
        # Bradley-Terry loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Accuracy: how often is chosen > rejected?
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        return loss, accuracy
    
    def train_step(self, chosen_ids, rejected_ids):
        """
        Single training step.
        
        Args:
            chosen_ids: Token IDs for chosen completions (batch_size, seq_len)
            rejected_ids: Token IDs for rejected completions (batch_size, seq_len)
        
        Returns:
            loss: Loss value
            accuracy: Accuracy value
        """
        self.model.train()
        
        # Move to device
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        
        # Forward pass
        chosen_rewards = self.model(chosen_ids)
        rejected_rewards = self.model(rejected_ids)
        
        # Compute loss
        loss, accuracy = self.compute_loss(chosen_rewards, rejected_rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), accuracy.item()
    
    @torch.no_grad()
    def eval_step(self, chosen_ids, rejected_ids):
        """
        Single evaluation step.
        
        Args:
            chosen_ids: Token IDs for chosen completions (batch_size, seq_len)
            rejected_ids: Token IDs for rejected completions (batch_size, seq_len)
        
        Returns:
            loss: Loss value
            accuracy: Accuracy value
        """
        self.model.eval()
        
        # Move to device
        chosen_ids = chosen_ids.to(self.device)
        rejected_ids = rejected_ids.to(self.device)
        
        # Forward pass
        chosen_rewards = self.model(chosen_ids)
        rejected_rewards = self.model(rejected_ids)
        
        # Compute loss
        loss, accuracy = self.compute_loss(chosen_rewards, rejected_rewards)
        
        return loss.item(), accuracy.item()


if __name__ == "__main__":
    # Test the reward model
    print("Testing Reward Model...")
    
    # Create a small config for testing
    config = GPTConfig(
        block_size=128,
        vocab_size=100,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False
    )
    
    # Create model
    model = RewardModel(config, pooling='last')
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    rewards = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {rewards.shape}")
    print(f"Sample rewards: {rewards}")
    
    # Test trainer
    print("\nTesting Trainer...")
    trainer = RewardModelTrainer(model, device='cpu')
    
    chosen_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    loss, acc = trainer.train_step(chosen_ids, rejected_ids)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    print("\n✅ Reward model test passed!")

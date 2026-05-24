"""
Utility functions for computing attention entropy.

Attention entropy is used to measure how concentrated or dispersed
the attention distribution is in Transformer models.
"""

import torch


def compute_attention_entropy(attention_probs, eps=1e-12):
    """
    Compute entropy for attention probability distributions.

    Args:
        attention_probs: Tensor of attention probabilities.
            Expected shape: [batch_size, num_heads, seq_len, seq_len]
        eps: Small value to avoid log(0).

    Returns:
        entropy: Tensor with entropy values.
            Shape: [batch_size, num_heads, seq_len]
    """
    attention_probs = attention_probs.clamp(min=eps)
    entropy = -torch.sum(attention_probs * torch.log(attention_probs), dim=-1)
    return entropy


def compute_mean_attention_entropy(attentions):
    """
    Compute mean attention entropy from all Transformer layers.

    Args:
        attentions: Tuple or list of attention tensors from Transformer outputs.
            Each tensor shape: [batch_size, num_heads, seq_len, seq_len]

    Returns:
        mean_entropy: Mean entropy value across layers, heads, and tokens.
    """
    entropy_values = []

    for layer_attention in attentions:
        layer_entropy = compute_attention_entropy(layer_attention)
        entropy_values.append(layer_entropy.mean())

    mean_entropy = torch.stack(entropy_values).mean()
    return mean_entropy


if __name__ == "__main__":
    dummy_attention = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    entropy = compute_attention_entropy(dummy_attention)
    mean_entropy = compute_mean_attention_entropy([dummy_attention])

    print("Entropy shape:", entropy.shape)
    print("Mean entropy:", mean_entropy.item())

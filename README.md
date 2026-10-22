# Neural Networks Learning

A hands-on exploration of neural networks and deep learning using Python and PyTorch. This repository documents my progression from implementing fundamental neural-network concepts to building CNNs, sequence models, attention mechanisms, and a GPT-style Transformer.

## Tiny GPT-Style Transformer

Built a character-level autoregressive Transformer from scratch in PyTorch to understand the core architecture behind modern language models.

The implementation includes:
- Character-level tokenization
- Token and positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Stacked Transformer blocks
- Next-token prediction
- Cross-entropy loss and AdamW optimization
- Autoregressive text generation
- Temperature-controlled sampling

The model is trained directly on raw text and learns to predict the next character in a sequence. During generation, it produces text autoregressively by sampling from predicted next-character probability distributions.

Experiments with different temperatures demonstrate the tradeoff between conservative, repetitive generation and more diverse but potentially noisy generation.

> This is intentionally a small educational implementation designed to understand the mechanics of GPT-style Transformer models rather than reproduce the scale or capabilities of modern large language models.
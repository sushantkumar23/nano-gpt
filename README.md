## Improving the Transformers

We will be building a GPT-like decoder-only transformer from scratch using PyTorch in phases, starting with the original Transformers introduced in the paper "Attention is All You Need" by Vaswani et al. We progressively move on to more advanced architectural improvements that have been proposed in recent research papers.

The transformer is implemented in `model.py` and the training can be done by setting the `training` flag to `True` in `model.py`.

### Original Transformer

- [x] Self-Attention
- [x] Scaled Dot-Product Attention
- [x] FeedForward Network
- [x] Absolute Positional Embedding
- [x] Residual Connection (Attention and FeedForward)
- [x] Layer Normalization (Attention and FeedForward)
- [x] Layer Normalization (Final)
- [x] Multi-Head Attention
- [x] Dropout

### Improvement over the years

- [ ] Rotary Positional Embedding
- [ ] RMS Layer Normalization
- [ ] KV Cache
- [ ] Multi-Query Attention
- [ ] SwiGLU Activation (FeedForward Network)
- [x] Flash Attention
- [ ] Sliding Window Attention
- [ ] Mixture of Experts

Each of the improvement were introduced over the years with a research paper.

#### RoPE: Rotary Positional Embedding
These were introduced in the paper [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

#### RMSNorm: Root Mean Square Layer Normalization
RMSNorms got introduced by Zhang et. al in 2019 in a paper called [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)


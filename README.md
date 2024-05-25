## Improving the Transformers

We will be building a GPT-like decoder-only transformer from scratch using PyTorch in phases, starting with the original Transformers introduced in the paper "Attention is All You Need" by Vaswani et al. We progressively move on to more advanced architectural improvements that have been proposed in recent research papers.

The transformer is implemented in `model.py` and the training can be done by setting the `training` flag to `True` in `model.py`.

### Phase 1

- [x] Self-Attention
- [x] Scaled Dot-Product Attention
- [x] FeedForward Network
- [x] Absolute Positional Embedding
- [x] Residual Connection (Attention and FeedForward)
- [x] Layer Normalization (Attention and FeedForward)
- [x] Layer Normalization (Final)
- [x] Multi-Head Attention
- [x] Dropout

### Phase 2

- [ ] Rotary Positional Embedding
- [ ] RMS Layer Normalization
- [ ] KV Cache
- [ ] Multi-Query Attention
- [ ] SwiGLU Activation (FeedForward Network)
- [ ] Flash Attention
- [ ] Sliding Window Attention
- [ ] Mixture of Experts

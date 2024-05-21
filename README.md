## Improving the Transformers

We will be building a GPT-like decoder-only transformer from scratch using PyTorch in phases, starting with the original Transformers introduced in the paper "Attention is All You Need" by Vaswani et al. and progressively move on to more advanced architectural improvements that have been proposed in recent research papers.

The transformer is implemented in `model.py` and the training can be done by setting the `training` flag to `True` in `model.py`.

### Phase 1

- [x] Self-Attention
- [x] Scaled Dot-Product Attention
- [x] Feed-Forward Network
- [x] Positional Embedding
- [x] Residual Connection
- [x] Layer Normalization
- [ ] Multi-Head Attention

### Phase 2

- [ ] Rotary Positional Embedding
- [ ] RMS Layer Normalization
- [ ] KV Cache
- [ ] Multi-Query Attention
- [ ] SwiGLU Activation (Feed-Forward Network)
- [ ] Sliding Window Attention
- [ ] Mixture of Experts

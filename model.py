# transfomer.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layers: int
    n_heads: int
    dim: int
    block_size: int
    ffn_multiplier: int = 4
    norm_eps: float = 1e-5


def get_batches(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(dataset[split], batch_size, model.config.block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads

        assert self.dim % self.n_heads == 0
        self.head_dim = config.dim // config.n_heads

        self.q_proj = nn.Linear(
            in_features=config.dim, out_features=self.n_heads * self.head_dim
        )
        self.k_proj = nn.Linear(
            in_features=config.dim, out_features=self.n_heads * self.head_dim
        )
        self.v_proj = nn.Linear(
            in_features=config.dim, out_features=self.n_heads * self.head_dim
        )

        self.out_proj = nn.Linear(
            in_features=self.n_heads * self.head_dim, out_features=config.dim
        )

        self.flash_attention = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.flash_attention:
            print("Using PyTorch's native attention")
            self.register_buffer(
                "mask", torch.tril(torch.ones(config.block_size, config.block_size))
            )

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, T, C) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # (B, n_heads, T, head_dim) x (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
            wei = q @ k.transpose(-2, -1) * (1.0 / (self.head_dim**0.5))
            wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)

            # (B, n_heads, T, T) x (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
            y = wei @ v

        output = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(output)

        return output


class SelfAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.dim
        self.wq = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.wk = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.wv = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Scaled Dot-Product Attention
        # (B, T, C) x (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (1.0 / (self.dim**0.5))
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        # (B, T, T) x (B, T, C) -> (B, T, C)
        output = wei @ v
        return output


class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ff1 = nn.Linear(
            in_features=config.dim, out_features=config.dim * config.ffn_multiplier
        )
        self.ff2 = nn.Linear(
            in_features=config.dim * config.ffn_multiplier, out_features=config.dim
        )

    def forward(self, x):
        return self.ff2(F.relu(self.ff1(x)))


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)

        # self.attention = SelfAttention(config)
        # MultiHeadSelfAttention: Drop-in replacement for SelfAttention
        self.attention = MultiHeadSelfAttention(config)

        self.feed_forward_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        residual = self.attention(self.attention_norm(x))
        x = x + residual

        residual = self.feed_forward(self.feed_forward_norm(x))
        x = x + residual
        return x


class LanguageModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.dim
        )
        self.pos_embedding_table = nn.Embedding(
            num_embeddings=config.block_size, embedding_dim=config.dim
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = nn.LayerNorm(config.dim, eps=config.norm_eps)

        self.lm_head = nn.Linear(
            in_features=config.dim, out_features=config.vocab_size
        )  # (B, T, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=x.device))

        # tok(B, T, C) + pos(T, C) [broadcasting] -> x(B, T, C)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            return logits, loss
        else:
            loss = None
        return logits, loss

    def generate(self, x, n):
        # x.shape = (B, T)
        for _ in range(n):
            # (B, T) -> (B, T, C)
            x_cond = (
                x
                if x.size(1) <= self.config.block_size
                else x[:, -self.config.block_size :]
            )
            logits, _ = self(x_cond)
            probs = F.softmax(logits, dim=-1)
            last_probs = probs[:, -1, :]
            idx_next = torch.multinomial(last_probs, 1)
            x = torch.cat([x, idx_next], dim=1)
        return x


if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Setting the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    # Setting the device to "mps" is causing issues in nn.Embedding
    print("Using {} device".format(device))

    # Data Preprocessing
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    itos = {i: c for i, c in enumerate(chars)}
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Initialization
    model_config = ModelConfig(
        vocab_size=vocab_size,
        n_layers=12,
        n_heads=8,
        dim=48,
        block_size=64,
    )
    model = LanguageModel(config=model_config).to(device)
    training = True

    # Training Code
    if training:

        # Train and Test Splits
        data = torch.tensor(encode(text), dtype=torch.long).to(device)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        dataset = {"train": train_data, "val": val_data}

        # Training Hyperparameters
        batch_size = 32
        max_iters = 5000
        eval_iters = 200
        eval_interval = 300
        learning_rate = 0.003

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for iter in range(max_iters):
            if iter % eval_interval == 0:
                losses = estimate_loss(
                    model, dataset, batch_size=batch_size, eval_iters=eval_iters
                )
                print(
                    f"iteration: {iter}, train_loss: {losses['train']:.4f} val_loss: {losses['val']:.4f}"
                )

            xb, yb = get_batches(
                train_data, batch_size=batch_size, block_size=model.config.block_size
            )
            logits, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # Autoregressive Inference
    start_idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    print(decode(model.generate(start_idx, n=600)[0].tolist()))

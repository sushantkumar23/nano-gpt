# transfomer.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int = 32
    block_size: int = 8


def get_batches(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, val_data, batch_size, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batches(val_data, batch_size, model.config.block_size)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out


class LanguageModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, x, targets=None):
        tok_emb = self.token_embedding_table(x)
        logits = self.lm_head(tok_emb)

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
            logits, _ = self(x)
            probs = F.softmax(logits, dim=-1)
            last_probs = probs[:, -1, :]
            idx_next = torch.multinomial(last_probs, 1)
            x = torch.cat([x, idx_next], dim=1)
        return x


if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Setting the device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using {} device".format(device))

    # Data Preprocessing
    chars = sorted(list(set(text)))
    itos = {i: c for i, c in enumerate(chars)}
    stoi = {c: i for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    # Train and Test Splits
    data = torch.tensor(encode(text), dtype=torch.long).to(device)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    vocab_size = len(chars)

    # Training Hyperparameters
    max_iters = 3000
    eval_iters = 200
    eval_interval = 300

    # Initialization
    model_config = ModelConfig(vocab_size=vocab_size)
    model = LanguageModel(config=model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(
                model, val_data, batch_size=32, eval_iters=eval_iters
            )
            print(f"Iteration: {iter}, Val Loss: {losses}")

        xb, yb = get_batches(train_data, batch_size=32, block_size=128)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Autoregressive Inference
    start_idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    print(decode(model.generate(start_idx, n=200)[0].tolist()))

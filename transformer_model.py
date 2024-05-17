# transfomer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Hyperparameters
    vocab_size = len(chars)
    n_embd = 32

    # Model Initialization
    m = LanguageModel(vocab_size=vocab_size, n_embd=n_embd)
    print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), n=200)[0].tolist()))

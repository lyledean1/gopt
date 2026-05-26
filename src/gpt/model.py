from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CausalSelfAttentionHead(nn.Module):
    """One attention head.

    Input shape:
    - x: (batch, time, d_model)

    Output shape:
    - out: (batch, time, head_size)
    """

    def __init__(self, *, d_model: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key_proj = nn.Linear(d_model, head_size, bias=False)
        self.query_proj = nn.Linear(d_model, head_size, bias=False)
        self.value_proj = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Lower-triangular mask so token t can only attend to tokens <= t.
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: Tensor) -> Tensor:
        _, time, _ = x.shape
        key = self.key_proj(x)
        query = self.query_proj(x)
        value = self.value_proj(x)

        # Compare every token with every earlier token.
        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores * (1.0 / math.sqrt(key.size(-1)))

        # Prevent the model from peeking at future tokens.
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:time, :time] == 0,
            float("-inf"),
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ value


class MultiHeadAttention(nn.Module):
    """Run several attention heads in parallel, then mix them back together."""

    def __init__(self, *, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        head_size = d_model // n_heads
        self.heads = nn.ModuleList(
            CausalSelfAttentionHead(
                d_model=d_model,
                head_size=head_size,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(n_heads)
        )
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        return self.dropout(self.proj(combined))


class FeedForward(nn.Module):
    """Per-token MLP applied after attention has mixed information."""

    def __init__(self, *, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: communicate with attention, then compute with an MLP."""

    def __init__(self, *, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model=d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Residual path 1: each token gathers information from earlier tokens.
        x = x + self.attention(self.attention_norm(x))

        # Residual path 2: each token transforms its own mixed representation.
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class GPTLanguageModel(nn.Module):
    """A tiny decoder-only transformer for next-token prediction."""

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.transformer_blocks = nn.Sequential(
            *[
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Predict logits for each token position.

        idx shape:
        - (batch, time)

        logits shape:
        - (batch, time, vocab_size)
        """

        batch_size, time = idx.shape
        if time > self.block_size:
            raise ValueError(f"sequence length {time} exceeds block size {self.block_size}")

        positions = torch.arange(time, device=idx.device)
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(positions)[None, :, :]

        # Each token starts as "what token am I?" + "where am I in the sequence?"
        hidden = token_embeddings + position_embeddings

        # Repeatedly mix information across positions and transform it.
        hidden = self.transformer_blocks(hidden)
        hidden = self.final_norm(hidden)

        # Project the final hidden state at each position to vocabulary scores.
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * time, -1),
                targets.reshape(batch_size * time),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int | None = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            # The model only knows how to look back block_size tokens.
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None and top_k > 0:
                k = min(top_k, next_token_logits.size(-1))
                top_values, _ = torch.topk(next_token_logits, k=k, dim=-1)
                cutoff = top_values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff,
                    float("-inf"),
                )
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

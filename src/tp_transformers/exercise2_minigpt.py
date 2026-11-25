"""Exercice 2 : Mini-GPT autorégressif (caractères) entraîné depuis zéro."""

from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tp_transformers.utils import count_parameters, get_device, set_seed


class CharTokenizer:
    """Tokeniseur par caractères avec tables stoi / itos."""

    def __init__(self, chars: Sequence[str]) -> None:
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Construit le vocabulaire trié à partir d'un corpus texte."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Nombre de caractères distincts du vocabulaire."""
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Convertit une chaîne en liste d'identifiants entiers."""
        return [self.stoi[c] for c in text]

    def decode(self, ids: Sequence[int] | torch.Tensor) -> str:
        """Reconvertit une séquence d'identifiants en chaîne de caractères."""
        if isinstance(ids, torch.Tensor):
            ids_list = [int(i) for i in ids.detach().cpu().tolist()]
        else:
            ids_list = [int(i) for i in ids]
        return "".join(self.itos[i] for i in ids_list)


class GPTBlock(nn.Module):
    """Bloc décodeur Pre-LN avec auto-attention causale multi-têtes et MLP GELU."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        masque = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=masque,
            need_weights=False,
            is_causal=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Modèle GPT minimal entraîné pour prédire le caractère suivant."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        block_size: int = 64,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [
                GPTBlock(
                    d_model=d_model,
                    nhead=nhead,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"Longueur de séquence ({seq_len}) > block_size ({self.block_size})."
            )
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Génère ``max_new_tokens`` caractères par échantillonnage autorégressif."""
        if temperature <= 0:
            raise ValueError("La température doit être strictement positive.")
        was_training = self.training
        self.eval()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=idx.device).manual_seed(seed)

        out = idx
        for _ in range(max_new_tokens):
            ctx = out[:, -self.block_size :]
            logits = self(ctx)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator)
            out = torch.cat([out, next_id], dim=1)

        if was_training:
            self.train()
        return out


@dataclass
class MiniGPTResult:
    """Résultats de l'entraînement, des vérifications et des générations MiniGPT."""

    vocab_size: int
    corpus_chars: int
    num_params: int
    causal_max_diff: float
    steps: list[int]
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float
    val_perplexity: float
    prompt: str
    generations: dict[float, str]
    elapsed_sec: float


def verify_causal_masking(
    model: MiniGPT,
    seq_len: int = 32,
    cut: int = 16,
    seed: int = 123,
) -> float:
    """Vérifie en eval() que modifier le futur n'altère pas les logits passés."""
    if not (1 <= cut < seq_len <= model.block_size):
        raise ValueError("Paramètres cut/seq_len invalides pour la vérification causale.")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    x = torch.randint(0, model.vocab_size, (2, seq_len), generator=gen, device=device)
    x_mod = x.clone()
    x_mod[:, cut:] = (x_mod[:, cut:] + 1) % model.vocab_size

    with torch.no_grad():
        logits_orig = model(x)
        logits_mod = model(x_mod)

    max_diff = float((logits_orig[:, :cut, :] - logits_mod[:, :cut, :]).abs().max().item())
    if was_training:
        model.train()
    return max_diff


def _sample_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def _estimate_loss(
    model: MiniGPT,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_batches: int,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(999)
    losses: list[float] = []
    for _ in range(eval_batches):
        xb, yb = _sample_batch(data, block_size, batch_size, device, gen)
        logits = model(xb)
        loss = criterion(logits.reshape(-1, model.vocab_size), yb.reshape(-1))
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train_minigpt(
    corpus_path: Path,
    block_size: int = 64,
    batch_size: int = 64,
    steps: int = 1200,
    eval_interval: int = 100,
    lr: float = 3e-3,
    seed: int = 42,
    prompt: str = "L'intelligence artificielle ",
    temperatures: tuple[float, ...] = (0.5, 1.0, 1.5),
    checkpoint_path: Path | None = None,
    plot_path: Path | None = None,
) -> MiniGPTResult:
    """Entraîne MiniGPT sur corpus.txt, vérifie la causalité et compare les températures."""
    set_seed(seed)
    device = get_device()
    text = corpus_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_idx = int(0.9 * len(tokens))
    train_data = tokens[:split_idx]
    val_data = tokens[split_idx:]

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        block_size=block_size,
        mlp_hidden=256,
        dropout=0.1,
    ).to(device)

    causal_max_diff = verify_causal_masking(model, seq_len=min(32, block_size), cut=16)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    train_gen = torch.Generator().manual_seed(seed)

    recorded_steps: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    t0 = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        xb, yb = _sample_batch(train_data, block_size, batch_size, device, train_gen)
        logits = model(xb)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), yb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % eval_interval == 0 or step == steps:
            tr_l = _estimate_loss(
                model, train_data, block_size, batch_size, 10, criterion, device
            )
            va_l = _estimate_loss(
                model, val_data, block_size, batch_size, 10, criterion, device
            )
            recorded_steps.append(step)
            train_losses.append(tr_l)
            val_losses.append(va_l)
            if va_l < best_val_loss:
                best_val_loss = va_l
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generations: dict[float, str] = {}
    for temp in temperatures:
        gen_ids = model.generate(
            prompt_ids,
            max_new_tokens=200,
            temperature=temp,
            seed=seed,
        )
        generations[temp] = tokenizer.decode(gen_ids[0])

    elapsed = time.perf_counter() - t0

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(recorded_steps, train_losses, label="Perte entraînement", linewidth=2)
        ax.plot(recorded_steps, val_losses, label="Perte validation", linewidth=2)
        ax.set_title("Mini-GPT : évolution de l'entropie croisée")
        ax.set_xlabel("Itération")
        ax.set_ylabel("CrossEntropyLoss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    return MiniGPTResult(
        vocab_size=tokenizer.vocab_size,
        corpus_chars=len(text),
        num_params=count_parameters(model),
        causal_max_diff=causal_max_diff,
        steps=recorded_steps,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        val_perplexity=float(math.exp(min(best_val_loss, 20.0))),
        prompt=prompt,
        generations=generations,
        elapsed_sec=elapsed,
    )

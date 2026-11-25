"""Exercice 3 : Mini-BERT bidirectionnel entraîné par Masked Language Modeling (MLM)."""

from __future__ import annotations

import copy
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tp_transformers.utils import count_parameters, get_device, set_seed

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
SPECIAL_TOKENS_COUNT = 5

SPECIAL_TOKEN_STRINGS = {
    PAD: "[PAD]",
    UNK: "[UNK]",
    CLS: "[CLS]",
    SEP: "[SEP]",
    MASK: "[MASK]",
}

_TOKEN_RE = re.compile(r"\[MASK\]|[a-zàâäéèêëïîôöùûüÿçœæ0-9]+|[.,!?;:'\"()-]", re.IGNORECASE)


def tokenize_words(text: str) -> list[str]:
    """Découpe une phrase en mots et ponctuations simples tout en préservant [MASK]."""
    raw_tokens = _TOKEN_RE.findall(text.strip())
    return ["[MASK]" if tok.upper() == "[MASK]" else tok.lower() for tok in raw_tokens]


class WordTokenizer:
    """Tokeniseur par mots avec identifiants spéciaux PAD..MASK = 0..4."""

    def __init__(self, max_len: int = 32) -> None:
        self.max_len = max_len
        self.stoi: dict[str, int] = {v: k for k, v in SPECIAL_TOKEN_STRINGS.items()}
        self.itos: dict[int, str] = dict(SPECIAL_TOKEN_STRINGS)

    @property
    def vocab_size(self) -> int:
        """Taille totale du vocabulaire (tokens spéciaux inclus)."""
        return len(self.stoi)

    def fit(self, sentences: Iterable[str]) -> WordTokenizer:
        """Construit le vocabulaire uniquement à partir des phrases d'entraînement."""
        vocab_words: set[str] = set()
        for sent in sentences:
            for tok in tokenize_words(sent):
                if tok not in SPECIAL_TOKEN_STRINGS.values():
                    vocab_words.add(tok)

        self.stoi = {v: k for k, v in SPECIAL_TOKEN_STRINGS.items()}
        for idx, word in enumerate(sorted(vocab_words), start=SPECIAL_TOKENS_COUNT):
            self.stoi[word] = idx
        self.itos = {i: s for s, i in self.stoi.items()}
        return self

    def encode(self, sentence: str) -> list[int]:
        """Encode sous la forme [CLS] mots... [SEP] [PAD]... de longueur max_len."""
        words = tokenize_words(sentence)
        max_content = self.max_len - 2
        words = words[:max_content]
        ids = [CLS] + [self.stoi.get(w, UNK) for w in words] + [SEP]
        if len(ids) < self.max_len:
            ids.extend([PAD] * (self.max_len - len(ids)))
        return ids

    def encode_batch(self, sentences: Sequence[str]) -> torch.Tensor:
        """Encode un lot de phrases en tenseur (N, max_len) de type torch.long."""
        return torch.tensor([self.encode(s) for s in sentences], dtype=torch.long)

    def decode(self, ids: Sequence[int] | torch.Tensor, skip_special: bool = True) -> str:
        """Reconvertit une séquence d'identifiants en texte lisible."""
        if isinstance(ids, torch.Tensor):
            id_list = [int(i) for i in ids.detach().cpu().tolist()]
        else:
            id_list = [int(i) for i in ids]
        tokens = [
            self.itos.get(i, "[UNK]")
            for i in id_list
            if not (skip_special and i in (PAD, CLS, SEP))
        ]
        return " ".join(tokens)


def apply_mlm_masking(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applique la corruption MLM (15 % sélectionnés : 80 % MASK, 10 % aléatoire, 10 % inchangé)."""
    if vocab_size <= SPECIAL_TOKENS_COUNT:
        raise ValueError("Le vocabulaire doit contenir au moins un token ordinaire.")

    device = input_ids.device
    ordinary = input_ids >= SPECIAL_TOKENS_COUNT

    rand_sel = torch.rand(input_ids.shape, generator=generator, device=device)
    selected = ordinary & (rand_sel < mask_prob)

    # Garantir au moins une position sélectionnée par phrase ayant des tokens ordinaires.
    for row in range(input_ids.size(0)):
        ord_positions = ordinary[row].nonzero(as_tuple=True)[0]
        if len(ord_positions) > 0 and not selected[row].any():
            pick_idx = int(
                torch.randint(0, len(ord_positions), (1,), generator=generator).item()
            )
            selected[row, ord_positions[pick_idx]] = True

    corrupted = input_ids.clone()
    rand_action = torch.rand(input_ids.shape, generator=generator, device=device)

    mask_positions = selected & (rand_action < 0.80)
    random_positions = selected & (rand_action >= 0.80) & (rand_action < 0.90)

    corrupted[mask_positions] = MASK
    num_rand = int(random_positions.sum().item())
    if num_rand > 0:
        corrupted[random_positions] = torch.randint(
            SPECIAL_TOKENS_COUNT,
            vocab_size,
            (num_rand,),
            generator=generator,
            device=device,
        )

    labels = input_ids.clone()
    labels[~selected] = -100
    return corrupted, labels, selected


class MiniBERT(nn.Module):
    """Encodeur Transformer bidirectionnel pour la modélisation du langage masqué."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Longueur {seq_len} > max_len {self.max_len}.")
        if src_key_padding_mask is None:
            src_key_padding_mask = input_ids == PAD
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos).unsqueeze(0)
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.head(encoded)


def verify_padding_invariance(
    model: MiniBERT,
    input_ids: torch.Tensor,
    seed: int = 123,
) -> float:
    """Vérifie en eval() que changer les valeurs de padding ne modifie pas les tokens utiles."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = input_ids.to(device)
    pad_mask = x == PAD
    useful_mask = ~pad_mask

    if not pad_mask.any() or not useful_mask.any():
        raise ValueError("Les séquences doivent contenir à la fois des tokens utiles et du PAD.")

    gen = torch.Generator(device=device).manual_seed(seed)
    x_mod = x.clone()
    x_mod[pad_mask] = torch.randint(
        SPECIAL_TOKENS_COUNT,
        model.vocab_size,
        (int(pad_mask.sum().item()),),
        generator=gen,
        device=device,
    )

    with torch.no_grad():
        logits_orig = model(x, src_key_padding_mask=pad_mask)
        logits_mod = model(x_mod, src_key_padding_mask=pad_mask)

    diff = (logits_orig[useful_mask] - logits_mod[useful_mask]).abs().max().item()
    if was_training:
        model.train()
    return float(diff)


@dataclass
class MiniBERTPrediction:
    """Top-k prédictions du MiniBERT pour une phrase contenant [MASK]."""

    sentence: str
    candidates: list[tuple[str, float]]


@dataclass
class MiniBERTResult:
    """Résultats complets de l'entraînement et de l'évaluation du MiniBERT."""

    vocab_size: int
    n_train: int
    n_val: int
    num_params: int
    padding_max_diff: float
    train_losses: list[float]
    val_losses: list[float]
    val_accuracies: list[float]
    best_val_accuracy: float
    best_val_loss: float
    sample_predictions: list[MiniBERTPrediction]
    elapsed_sec: float


@torch.no_grad()
def predict_top_k(
    model: MiniBERT,
    tokenizer: WordTokenizer,
    masked_sentence: str,
    k: int = 3,
) -> MiniBERTPrediction:
    """Prédit les k mots les plus probables pour la position [MASK] d'une phrase."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(masked_sentence)], dtype=torch.long, device=device)
    mask_positions = (ids[0] == MASK).nonzero(as_tuple=True)[0]
    if len(mask_positions) != 1:
        raise ValueError(
            f"La phrase doit contenir exactement un [MASK] (trouvé : {len(mask_positions)})."
        )
    mask_pos = int(mask_positions[0].item())
    logits = model(ids)
    probs = torch.softmax(logits[0, mask_pos, :], dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)
    candidates = [
        (tokenizer.itos.get(int(idx.item()), "[UNK]"), float(p.item()))
        for p, idx in zip(top_probs, top_ids)
    ]
    return MiniBERTPrediction(sentence=masked_sentence, candidates=candidates)


def train_minibert(
    corpus_path: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-3,
    max_len: int = 32,
    seed: int = 42,
    test_prompts: Sequence[str] | None = None,
    checkpoint_path: Path | None = None,
    plot_path: Path | None = None,
) -> MiniBERTResult:
    """Entraîne MiniBERT sur un corpus de phrases et évalue l'exactitude MLM."""
    set_seed(seed)
    device = get_device()
    lines = [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Mélange reproductible puis séparation train / val (85 % / 15 %)
    perm = torch.randperm(len(lines), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [lines[i] for i in perm]
    split = int(0.85 * len(shuffled))
    train_sentences = shuffled[:split]
    val_sentences = shuffled[split:]

    tokenizer = WordTokenizer(max_len=max_len).fit(train_sentences)
    train_ids = tokenizer.encode_batch(train_sentences)
    val_ids = tokenizer.encode_batch(val_sentences)

    # Jeu de validation masqué de manière déterministe pour suivre la progression
    val_gen = torch.Generator().manual_seed(seed + 100)
    val_corrupted, val_labels, val_selected = apply_mlm_masking(
        val_ids, tokenizer.vocab_size, mask_prob=0.15, generator=val_gen
    )

    model = MiniBERT(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_len=max_len,
        dropout=0.1,
    ).to(device)

    padding_max_diff = verify_padding_invariance(model, val_ids[:4])

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    train_loader = DataLoader(TensorDataset(train_ids), batch_size=batch_size, shuffle=True)
    mlm_gen = torch.Generator().manual_seed(seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0
        for (batch_ids,) in train_loader:
            corrupted, labels, _ = apply_mlm_masking(
                batch_ids, tokenizer.vocab_size, mask_prob=0.15, generator=mlm_gen
            )
            corrupted = corrupted.to(device)
            labels = labels.to(device)

            logits = model(corrupted)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += float(loss.item())
            total_batches += 1

        train_losses.append(running_loss / max(total_batches, 1))

        model.eval()
        with torch.no_grad():
            v_logits = model(val_corrupted.to(device))
            v_loss = criterion(
                v_logits.reshape(-1, tokenizer.vocab_size),
                val_labels.to(device).reshape(-1),
            )
            preds = v_logits.argmax(dim=-1).cpu()
            correct = (preds[val_selected] == val_ids[val_selected]).float().mean().item()

        v_loss_f = float(v_loss.item())
        val_losses.append(v_loss_f)
        val_accuracies.append(float(correct))

        if v_loss_f < best_val_loss:
            best_val_loss = v_loss_f
            best_val_acc = float(correct)
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    prompts = test_prompts or [
        "le chat dort sur le [MASK] rouge",
        "rabat est la [MASK] du maroc",
        "le médecin utilise un [MASK] précis",
        "en hiver la [MASK] tombe sur la montagne",
    ]
    sample_predictions = [predict_top_k(model, tokenizer, p, k=3) for p in prompts]
    elapsed = time.perf_counter() - t0

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(train_losses, label="Perte entraînement", linewidth=2)
        axes[0].plot(val_losses, label="Perte validation", linewidth=2)
        axes[0].set_title("Mini-BERT : perte MLM")
        axes[0].set_xlabel("Époque")
        axes[0].set_ylabel("CrossEntropyLoss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot([a * 100 for a in val_accuracies], color="#2ca02c", linewidth=2)
        axes[1].set_title("Mini-BERT : exactitude validation (%)")
        axes[1].set_xlabel("Époque")
        axes[1].set_ylabel("Exactitude (%)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    return MiniBERTResult(
        vocab_size=tokenizer.vocab_size,
        n_train=len(train_sentences),
        n_val=len(val_sentences),
        num_params=count_parameters(model),
        padding_max_diff=padding_max_diff,
        train_losses=train_losses,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
        best_val_accuracy=best_val_acc,
        best_val_loss=best_val_loss,
        sample_predictions=sample_predictions,
        elapsed_sec=elapsed,
    )

"""Tests unitaires et d'invariance causale pour l'Exercice 2 (MiniGPT)."""

from __future__ import annotations

import pytest
import torch

from tp_transformers.exercise2_minigpt import (
    CharTokenizer,
    MiniGPT,
    verify_causal_masking,
)


def test_char_tokenizer_roundtrip() -> None:
    text = "Bonjour le monde !"
    tok = CharTokenizer.from_text(text)
    ids = tok.encode("Bonjour !")
    assert tok.decode(ids) == "Bonjour !"
    assert tok.vocab_size == len(set(text))


def test_minigpt_shapes_and_causal_masking() -> None:
    model = MiniGPT(vocab_size=30, d_model=64, nhead=4, num_layers=2, block_size=64)
    x = torch.randint(0, 30, (4, 64))
    logits = model(x)
    assert logits.shape == (4, 64, 30)

    with pytest.raises(ValueError):
        model(torch.randint(0, 30, (1, 65)))

    # Vérification essentielle : modifier un caractère futur ne change pas les logits passés
    max_diff = verify_causal_masking(model, seq_len=32, cut=16)
    assert max_diff < 1e-6


def test_minigpt_generation_length_and_temperature_entropy() -> None:
    model = MiniGPT(vocab_size=25, d_model=64, nhead=4, num_layers=2, block_size=64)
    prompt = torch.randint(0, 25, (1, 5))
    out = model.generate(prompt, max_new_tokens=200, temperature=1.0, seed=42)
    assert out.shape == (1, 205)

    # La température accroît l'entropie de la distribution softmax
    logits = torch.tensor([3.0, 1.0, 0.2, -1.0, -2.5])
    entropies = []
    for temp in (0.5, 1.0, 1.5):
        p = torch.softmax(logits / temp, dim=-1)
        entropy = float(-(p * torch.log(p)).sum().item())
        entropies.append(entropy)
    assert entropies[0] < entropies[1] < entropies[2]

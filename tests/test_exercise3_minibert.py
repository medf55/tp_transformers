"""Tests unitaires et d'invariance au padding pour l'Exercice 3 (MiniBERT)."""

from __future__ import annotations

import torch

from tp_transformers.exercise3_minibert import (
    CLS,
    MASK,
    PAD,
    SEP,
    SPECIAL_TOKENS_COUNT,
    UNK,
    MiniBERT,
    WordTokenizer,
    apply_mlm_masking,
    verify_padding_invariance,
)


def test_word_tokenizer_special_ids_and_oov() -> None:
    assert (PAD, UNK, CLS, SEP, MASK) == (0, 1, 2, 3, 4)

    tok = WordTokenizer(max_len=32).fit(["le chat dort sur le canapé"])
    encoded = tok.encode("le chat dort sur un tapis")
    assert len(encoded) == 32
    assert encoded[0] == CLS
    assert encoded[7] == SEP
    assert all(t == PAD for t in encoded[8:])
    # "un" et "tapis" n'étaient pas dans le corpus d'entraînement
    assert encoded[5] == UNK
    assert encoded[6] == UNK


def test_mlm_masking_rules_and_ratios() -> None:
    vocab_size = 50
    # 200 séquences de 12 tokens ordinaires + CLS/SEP + PAD
    rows = []
    for _ in range(200):
        seq = [CLS] + list(range(5, 17)) + [SEP] + [PAD] * 18
        rows.append(seq)
    batch = torch.tensor(rows, dtype=torch.long)

    gen = torch.Generator().manual_seed(42)
    corrupted, labels, selected = apply_mlm_masking(batch, vocab_size=vocab_size, generator=gen)

    # Aucun token spécial ne doit être sélectionné
    special_mask = batch < SPECIAL_TOKENS_COUNT
    assert not (selected & special_mask).any()

    # Au moins une position sélectionnée par phrase
    assert (selected.sum(dim=1) >= 1).all()

    # Hors sélection : label = -100 et token non modifié
    assert (labels[~selected] == -100).all()
    assert torch.equal(corrupted[~selected], batch[~selected])
    assert torch.equal(labels[selected], batch[selected])

    # Vérification des proportions ~ 80 % MASK / 10 % random / 10 % inchangé
    n_sel = int(selected.sum().item())
    n_mask = int((corrupted[selected] == MASK).sum().item())
    n_unchanged = int((corrupted[selected] == batch[selected]).sum().item())
    assert 0.65 <= n_mask / n_sel <= 0.90
    assert 0.03 <= n_unchanged / n_sel <= 0.25


def test_minibert_padding_invariance() -> None:
    tok = WordTokenizer(max_len=32).fit(
        ["le chat dort sur le canapé rouge", "rabat est la capitale du maroc"]
    )
    batch = tok.encode_batch(
        ["le chat dort", "rabat est la capitale du maroc"]
    )
    model = MiniBERT(vocab_size=tok.vocab_size, d_model=64, nhead=4, num_layers=2, max_len=32)

    logits = model(batch)
    assert logits.shape == (2, 32, tok.vocab_size)

    # Vérification essentielle : modifier les positions de padding ne change pas les tokens utiles
    max_diff = verify_padding_invariance(model, batch)
    assert max_diff < 1e-6

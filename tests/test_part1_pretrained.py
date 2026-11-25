"""Tests pour les modèles préentraînés GPT-2 et BERT (Première Partie)."""

from __future__ import annotations

import os

import pytest

from tp_transformers.part1_pretrained import (
    BERTMaskPredictor,
    GPT2Generator,
    build_offline_demo_bert,
    build_offline_demo_gpt2,
)


def test_gpt2_and_bert_pipeline_offline() -> None:
    gpt = build_offline_demo_gpt2()
    gpt_res = gpt.generate(
        prompt="Artificial intelligence can help people",
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9,
        seed=42,
    )
    assert gpt_res.input_shape == (1, 5)
    assert 1 <= gpt_res.num_new_tokens <= 10
    assert gpt_res.generated_text.startswith("Artificial intelligence can help people")
    assert gpt.tokenizer.pad_token_id == gpt.tokenizer.eos_token_id

    bert = build_offline_demo_bert()
    with pytest.raises(ValueError):
        bert.predict_masked("Phrase sans masque.")
    with pytest.raises(ValueError):
        bert.predict_masked("Deux [MASK] dans la [MASK] phrase.")

    bert_res = bert.predict_masked("Rabat est la [MASK] du Maroc.", top_k=5)
    assert len(bert_res.candidates) == 5
    assert bert_res.mask_position >= 0
    probs = [c.probability for c in bert_res.candidates]
    assert all(0.0 < p <= 1.0 for p in probs)
    assert sum(probs) <= 1.0001


@pytest.mark.skipif(
    os.getenv("TP_RUN_LIVE_HF") != "1",
    reason="Téléchargement 1,26 Go (gpt2 + bert-base-multilingual-cased) désactivé par défaut",
)
def test_gpt2_generation_and_bert_mask_prediction_live() -> None:
    gpt = GPT2Generator("gpt2")
    gpt_res = gpt.generate(
        prompt="Artificial intelligence can help people",
        max_new_tokens=15,
        temperature=0.8,
        top_p=0.9,
        seed=42,
    )
    assert gpt_res.input_shape == (1, 6)
    assert gpt_res.num_new_tokens > 0
    assert gpt_res.generated_text.startswith("Artificial intelligence can help people")

    bert = BERTMaskPredictor("bert-base-multilingual-cased")
    bert_res = bert.predict_masked("Rabat est la [MASK] du Maroc.", top_k=5)
    top3_tokens = [c.token.lower() for c in bert_res.candidates[:3]]
    assert "capitale" in top3_tokens

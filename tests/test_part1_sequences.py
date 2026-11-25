"""Tests unitaires et de convergence pour la Première Partie (RNN, LSTM, Transformer)."""

from __future__ import annotations

import torch

from tp_transformers.part1_sequences import (
    MiniTransformer,
    SimpleLSTM,
    SimpleRNN,
    create_sine_dataset,
    run_part1_sequence_benchmark,
)


def test_create_sine_dataset_shapes_and_chronological_split() -> None:
    ds = create_sine_dataset(n_points=200, x_max=20.0, seq_len=10, train_size=150)

    assert ds.periode.shape == (200,)
    assert ds.serie.shape == (200,)
    assert ds.X_train.shape == (150, 10, 1)
    assert ds.y_train.shape == (150, 1)
    assert ds.X_test.shape == (40, 10, 1)
    assert ds.y_test.shape == (40, 1)

    # La cible du dernier exemple train correspond au dernier élément de la première fenêtre test + 1 pas
    assert torch.allclose(ds.y_train[-1, 0], ds.X_test[0, -1, 0])


def test_sequence_models_forward_shapes() -> None:
    x = torch.randn(8, 10, 1)
    for model in [SimpleRNN(), SimpleLSTM(), MiniTransformer()]:
        out = model(x)
        assert out.shape == (8, 1)


def test_part1_models_converge_on_sine() -> None:
    _, results = run_part1_sequence_benchmark(epochs=200, lr=0.01, seed=42)
    assert len(results) == 3
    for res in results:
        assert res.test_mse < 0.01, f"{res.name} n'a pas convergé (MSE={res.test_mse})"
        assert res.train_losses[-1] < res.train_losses[0]

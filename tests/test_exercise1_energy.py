"""Tests unitaires pour l'Exercice 1 (EnergyTransformer et préparation sans fuite)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tp_transformers.exercise1_energy import (
    EnergyTransformer,
    prepare_energy_data,
    train_and_evaluate_energy,
)


def _write_synthetic_power_csv(
    path: Path,
    n_hours: int = 100,
    sparse_hour_idx: int | None = 15,
    test_scale: float = 1.0,
) -> None:
    """Crée un fichier minute synthétique au format UCI semicolon."""
    start = pd.Timestamp("2020-01-01 00:00:00")
    fin_train_hour = int(0.70 * n_hours)
    rows: list[str] = ["Date;Time;Global_active_power"]

    for h in range(n_hours):
        base_val = 1.5 + 0.5 * np.sin(2 * np.pi * h / 24.0)
        if h >= int(0.85 * n_hours):
            base_val *= test_scale
        for m in range(60):
            ts = start + pd.Timedelta(hours=h, minutes=m)
            date_s = ts.strftime("%d/%m/%Y")
            time_s = ts.strftime("%H:%M:%S")
            if sparse_hour_idx is not None and h == sparse_hour_idx and m >= 30:
                val_s = "?"
            else:
                val_s = f"{base_val + 0.01 * (m % 5):.3f}"
            rows.append(f"{date_s};{time_s};{val_s}")

    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_prepare_energy_data_filtering_and_no_leakage(tmp_path: Path) -> None:
    csv_a = tmp_path / "energy_a.txt"
    csv_b = tmp_path / "energy_b.txt"
    _write_synthetic_power_csv(csv_a, n_hours=120, sparse_hour_idx=15, test_scale=1.0)
    _write_synthetic_power_csv(csv_b, n_hours=120, sparse_hour_idx=15, test_scale=50.0)

    bundle_a = prepare_energy_data(csv_a, window_size=24, min_count=45, batch_size=16)
    bundle_b = prepare_energy_data(csv_b, window_size=24, min_count=45, batch_size=16)

    # Normalisation identique malgré la perturbation du jeu de test : aucune fuite de données
    assert bundle_a.mean == pytest.approx(bundle_b.mean, rel=1e-6)
    assert bundle_a.std == pytest.approx(bundle_b.std, rel=1e-6)

    # L'heure 15 a seulement 30 mesures (< 45), donc toutes les fenêtres qui la touchent sont écartées
    assert bundle_a.n_hours == 120
    assert bundle_a.n_train + bundle_a.n_val + bundle_a.n_test < 120 - 24
    assert torch.isfinite(bundle_a.X_test).all()
    assert torch.isfinite(bundle_a.y_test).all()


def test_energy_transformer_shapes_and_training(tmp_path: Path) -> None:
    model = EnergyTransformer()
    x = torch.randn(4, 24, 1)
    assert model(x).shape == (4, 1)

    csv_path = tmp_path / "energy.txt"
    _write_synthetic_power_csv(csv_path, n_hours=160, sparse_hour_idx=None, test_scale=1.0)
    bundle = prepare_energy_data(csv_path, window_size=24, min_count=45, batch_size=16)
    plot_path = tmp_path / "energy_plot.png"

    res = train_and_evaluate_energy(
        bundle,
        max_epochs=2,
        lr=1e-3,
        seed=42,
        plot_path=plot_path,
    )
    assert len(res.train_losses) == 2
    assert len(res.val_losses) == 2
    assert plot_path.exists()
    assert res.transformer_metrics_kw["mae"] >= 0.0
    assert res.persistence_metrics_kw["mae"] >= 0.0
    assert res.daily_metrics_kw["mae"] >= 0.0

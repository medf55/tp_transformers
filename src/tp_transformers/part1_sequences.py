"""Première Partie : prédiction de série temporelle sinusoïdale (RNN, LSTM, Transformer)."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tp_transformers.utils import (
    compute_regression_metrics,
    count_parameters,
    set_seed,
)


@dataclass(frozen=True)
class SineDataset:
    """Jeu de données séquentiel généré à partir de y = sin(x)."""

    periode: torch.Tensor
    serie: torch.Tensor
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


@dataclass
class SequenceResult:
    """Résultats d'entraînement et d'évaluation pour un modèle séquentiel."""

    name: str
    num_params: int
    train_losses: list[float]
    test_mse: float
    test_rmse: float
    test_mae: float
    predictions: np.ndarray
    elapsed_sec: float


def create_sine_dataset(
    n_points: int = 200,
    x_max: float = 20.0,
    seq_len: int = 10,
    train_size: int = 150,
) -> SineDataset:
    """Génère y = sin(x) et construit des fenêtres glissantes chronologiques."""
    if n_points <= seq_len:
        raise ValueError("n_points doit être strictement supérieur à seq_len.")
    total_samples = n_points - seq_len
    if not (0 < train_size < total_samples):
        raise ValueError("train_size doit être compris entre 1 et n_points - seq_len - 1.")

    periode = torch.linspace(0.0, x_max, n_points)
    serie = torch.sin(periode)

    X = torch.stack([serie[i : i + seq_len] for i in range(total_samples)]).unsqueeze(-1)
    y = torch.stack([serie[i + seq_len] for i in range(total_samples)]).unsqueeze(-1)

    return SineDataset(
        periode=periode,
        serie=serie,
        X_train=X[:train_size],
        y_train=y[:train_size],
        X_test=X[train_size:],
        y_test=y[train_size:],
    )


class SimpleRNN(nn.Module):
    """Modèle RNN simple prédisant le pas suivant à partir du dernier état caché."""

    def __init__(self, input_size: int = 1, hidden_size: int = 16) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.sortie = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        etats, _ = self.rnn(x)
        dernier_etat = etats[:, -1, :]
        return self.sortie(dernier_etat)


class SimpleLSTM(nn.Module):
    """Modèle LSTM simple prédisant le pas suivant à partir du dernier état caché."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.sortie = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        etats, _ = self.lstm(x)
        dernier_etat = etats[:, -1, :]
        return self.sortie(dernier_etat)


class MiniTransformer(nn.Module):
    """Encodeur Transformer léger pour la prédiction de série temporelle."""

    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 2,
        dim_feedforward: int = 32,
        num_layers: int = 1,
        max_len: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        self.positions = nn.Embedding(max_len, d_model)
        couche = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encodeur = nn.TransformerEncoder(
            couche,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.sortie = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        longueur = x.size(1)
        proj = self.projection(x)
        indices = torch.arange(longueur, device=x.device)
        positions = self.positions(indices).unsqueeze(0)
        representations = self.encodeur(proj + positions)
        derniere_position = representations[:, -1, :]
        return self.sortie(derniere_position)


def train_sequence_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.01,
    seed: int = 42,
    name: str | None = None,
) -> SequenceResult:
    """Entraîne un modèle séquentiel avec MSE + Adam et évalue sur le jeu de test."""
    set_seed(seed)
    critere = nn.MSELoss()
    optimiseur = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []

    t0 = time.perf_counter()
    model.train()
    for _ in range(epochs):
        predictions = model(X_train)
        perte = critere(predictions, y_train)
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        train_losses.append(float(perte.item()))

    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
    elapsed = time.perf_counter() - t0

    metrics = compute_regression_metrics(y_test, test_preds)
    return SequenceResult(
        name=name or model.__class__.__name__,
        num_params=count_parameters(model),
        train_losses=train_losses,
        test_mse=metrics["mse"],
        test_rmse=metrics["rmse"],
        test_mae=metrics["mae"],
        predictions=test_preds.detach().cpu().numpy().reshape(-1),
        elapsed_sec=elapsed,
    )


def plot_part1_comparison(
    dataset: SineDataset,
    results: list[SequenceResult],
    save_path: Path,
) -> None:
    """Trace les courbes d'apprentissage et les prédictions de test des 3 modèles."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = dataset.y_test.detach().cpu().numpy().reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for res in results:
        axes[0].plot(res.train_losses, label=f"{res.name} ({res.num_params} params)")
    axes[0].set_yscale("log")
    axes[0].set_title("Évolution de la perte MSE (échelle log)")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("MSE entraînement")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(y_true, label="Valeurs réelles", color="black", linewidth=2.2)
    styles = ["--", "-.", ":"]
    for res, ls in zip(results, styles):
        axes[1].plot(
            res.predictions,
            linestyle=ls,
            linewidth=2,
            label=f"{res.name} (RMSE={res.test_rmse:.4f})",
        )
    axes[1].set_title("Prédictions sur le jeu de test (40 points)")
    axes[1].set_xlabel("Indice de test")
    axes[1].set_ylabel("sin(x)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_part1_sequence_benchmark(
    epochs: int = 300,
    lr: float = 0.01,
    seed: int = 42,
    save_plot_path: Path | None = None,
) -> tuple[SineDataset, list[SequenceResult]]:
    """Exécute le benchmark complet RNN vs LSTM vs MiniTransformer sur y = sin(x)."""
    dataset = create_sine_dataset()

    builders = [
        ("SimpleRNN", lambda: SimpleRNN(input_size=1, hidden_size=16)),
        ("SimpleLSTM", lambda: SimpleLSTM(input_size=1, hidden_size=16, num_layers=1)),
        (
            "MiniTransformer",
            lambda: MiniTransformer(
                d_model=16,
                nhead=2,
                dim_feedforward=32,
                num_layers=1,
                max_len=10,
                dropout=0.0,
            ),
        ),
    ]

    results: list[SequenceResult] = []
    for name, builder in builders:
        set_seed(seed)
        model = builder()
        res = train_sequence_model(
            model=model,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            y_test=dataset.y_test,
            epochs=epochs,
            lr=lr,
            seed=seed,
            name=name,
        )
        results.append(res)

    if save_plot_path is not None:
        plot_part1_comparison(dataset, results, save_plot_path)

    return dataset, results

"""Utilitaires communs : reproductibilité, matériel, métriques et environnement HuggingFace."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def configure_hf_environment() -> None:
    """Configure les variables d'environnement HuggingFace Hub avant tout import.

    Désactive ``hf-xet`` sous Windows pour éviter les blocages lors du
    téléchargement de poids depuis HuggingFace Hub.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def set_seed(seed: int = 42) -> None:
    """Fixe les graines aléatoires Python, NumPy et PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Retourne le périphérique de calcul disponible (CUDA si présent, sinon CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    """Retourne le nombre total de paramètres entraînables d'un modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_regression_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    """Calcule MSE, RMSE et MAE entre valeurs réelles et prédictions."""
    if isinstance(y_true, torch.Tensor):
        y_true_arr = y_true.detach().cpu().numpy().reshape(-1)
    else:
        y_true_arr = np.asarray(y_true, dtype=np.float64).reshape(-1)

    if isinstance(y_pred, torch.Tensor):
        y_pred_arr = y_pred.detach().cpu().numpy().reshape(-1)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    diff = y_pred_arr - y_true_arr
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    return {"mse": mse, "rmse": rmse, "mae": mae}

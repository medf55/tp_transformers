"""Exercice 1 : Prédiction de la consommation électrique horaire avec un Transformer."""

from __future__ import annotations

import copy
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tp_transformers.utils import (
    compute_regression_metrics,
    count_parameters,
    get_device,
    set_seed,
)

UCI_ENERGY_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


@dataclass
class EnergyDatasetBundle:
    """Conteneur des DataLoaders et métadonnées pour la consommation électrique."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    X_test: torch.Tensor
    y_test: torch.Tensor
    test_timestamps: np.ndarray
    mean: float
    std: float
    n_hours: int
    n_train: int
    n_val: int
    n_test: int


@dataclass
class EnergyResult:
    """Résultats complets de l'entraînement et de l'évaluation sur UCI Energy."""

    num_params: int
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    transformer_metrics_kw: dict[str, float]
    persistence_metrics_kw: dict[str, float]
    daily_metrics_kw: dict[str, float]
    y_test_kw: np.ndarray
    preds_kw: np.ndarray
    persistence_kw: np.ndarray
    daily_kw: np.ndarray
    test_timestamps: np.ndarray
    elapsed_sec: float


def download_energy_dataset(data_dir: Path) -> Path:
    """Télécharge et extrait le jeu de données UCI de manière idempotente."""
    data_dir.mkdir(parents=True, exist_ok=True)
    txt_path = data_dir / "household_power_consumption.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
        return txt_path

    zip_path = data_dir / "household_power_consumption.zip"
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        req = urllib.request.Request(
            UCI_ENERGY_URL,
            headers={"User-Agent": "tp-transformers/0.1"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith("household_power_consumption.txt")]
        if not members:
            raise FileNotFoundError(
                "household_power_consumption.txt introuvable dans l'archive UCI."
            )
        with zf.open(members[0]) as src, open(txt_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 256)
                if not chunk:
                    break
                dst.write(chunk)

    return txt_path


def generate_realistic_energy_fallback(
    output_path: Path,
    n_days: int = 365,
    seed: int = 42,
) -> Path:
    """Génère un jeu minute réaliste au format UCI lorsque le téléchargement externe est bloqué.

    Combine cycle diurne (pics matin/soir), profil week-end, saisonnalité annuelle,
    processus autorégressif AR(1) et quelques heures incomplètes (``?``).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    rng = np.random.default_rng(seed)
    n_minutes = n_days * 24 * 60
    timestamps = pd.date_range("2009-01-01 00:00:00", periods=n_minutes, freq="min")

    hour = timestamps.hour.to_numpy(dtype=np.float64) + timestamps.minute.to_numpy(dtype=np.float64) / 60.0
    dayofweek = timestamps.dayofweek.to_numpy(dtype=np.float64)
    dayofyear = timestamps.dayofyear.to_numpy(dtype=np.float64)

    # Profil diurne réaliste (creux nocturne, pic matin ~8h, pic soir ~20h)
    diurnal = (
        0.55
        + 0.65 * np.exp(-0.5 * ((hour - 8.0) / 1.8) ** 2)
        + 1.05 * np.exp(-0.5 * ((hour - 20.0) / 2.2) ** 2)
        + 0.25 * np.exp(-0.5 * ((hour - 13.0) / 1.5) ** 2)
    )
    weekend_shift = np.where(dayofweek >= 5, 0.20, 0.0)
    seasonal = 0.35 * np.cos(2.0 * np.pi * (dayofyear - 15.0) / 365.0)

    # Bruit AR(1) horaire lissé + petite fluctuation minute
    n_hours = n_days * 24
    ar_hourly = np.zeros(n_hours, dtype=np.float64)
    eps = rng.normal(0.0, 0.14, size=n_hours)
    for t in range(1, n_hours):
        ar_hourly[t] = 0.72 * ar_hourly[t - 1] + eps[t]
    ar_min = np.repeat(ar_hourly, 60)
    noise_min = rng.normal(0.0, 0.06, size=n_minutes)

    power = np.clip(diurnal + weekend_shift + seasonal + ar_min + noise_min, 0.12, 8.0)
    power_str = np.char.mod("%.3f", power)

    # Introduire 3 heures partiellement manquantes (< 45 mesures valides) pour exercer le filtrage
    for sparse_hour in (40, 1800, 5200):
        start_idx = sparse_hour * 60 + 25
        end_idx = (sparse_hour + 1) * 60
        if end_idx <= n_minutes:
            power_str[start_idx:end_idx] = "?"

    df = pd.DataFrame(
        {
            "Date": timestamps.strftime("%d/%m/%Y"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Global_active_power": power_str,
        }
    )
    df.to_csv(output_path, sep=";", index=False)
    return output_path


def prepare_energy_data(
    file_path: Path,
    window_size: int = 24,
    min_count: int = 45,
    batch_size: int = 64,
) -> EnergyDatasetBundle:
    """Agrège par heure, normalise sans fuite et découpe en train/val/test."""
    df = pd.read_csv(
        file_path,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        na_values=["?"],
        low_memory=False,
    )

    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    puissance = pd.to_numeric(df["Global_active_power"], errors="coerce")
    puissance.index = pd.DatetimeIndex(dt)

    horaire = puissance.resample("h").agg(["mean", "count"])
    serie = horaire["mean"].where(horaire["count"] >= min_count)

    n = len(serie)
    fin_train = int(0.70 * n)
    fin_val = int(0.85 * n)

    train_slice = serie.iloc[:fin_train]
    mean = float(train_slice.mean())
    std = float(train_slice.std())
    if std <= 0 or not np.isfinite(std):
        raise ValueError("Écart-type d'entraînement invalide pour la normalisation.")

    valeurs = ((serie - mean) / std).to_numpy(dtype=np.float32)
    index_arr = serie.index.to_numpy()

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    dates_cibles: list[int] = []
    ts_list: list[np.datetime64] = []

    for i in range(window_size, n):
        sequence = valeurs[i - window_size : i]
        cible = valeurs[i]
        if np.isfinite(sequence).all() and np.isfinite(cible):
            X_list.append(sequence)
            y_list.append(float(cible))
            dates_cibles.append(i)
            ts_list.append(index_arr[i])

    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)
    dates_arr = np.array(dates_cibles)
    ts_arr = np.array(ts_list)

    train_mask = dates_arr < fin_train
    val_mask = (dates_arr >= fin_train) & (dates_arr < fin_val)
    test_mask = dates_arr >= fin_val

    train_loader = DataLoader(
        TensorDataset(X[train_mask], y[train_mask]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X[val_mask], y[val_mask]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X[test_mask], y[test_mask]),
        batch_size=batch_size,
        shuffle=False,
    )

    return EnergyDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        X_test=X[test_mask],
        y_test=y[test_mask],
        test_timestamps=ts_arr[test_mask],
        mean=mean,
        std=std,
        n_hours=n,
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
        n_test=int(test_mask.sum()),
    )


class EnergyTransformer(nn.Module):
    """Transformer encodeur pour prédire la consommation de l'heure suivante."""

    def __init__(
        self,
        window_size: int = 24,
        d_model: int = 32,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(1, d_model)
        self.positions = nn.Embedding(window_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        h = self.projection(x)
        pos_idx = torch.arange(seq_len, device=x.device)
        h = h + self.positions(pos_idx).unsqueeze(0)
        h = self.encoder(h)
        return self.head(h[:, -1, :])


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Calcule la perte MSE moyenne et rassemble les prédictions sur un DataLoader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds_batches: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            bs = xb.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            preds_batches.append(pred.cpu().numpy().reshape(-1))

    mean_loss = total_loss / max(total_samples, 1)
    all_preds = np.concatenate(preds_batches) if preds_batches else np.empty(0, dtype=np.float32)
    return mean_loss, all_preds


def _find_contiguous_window(timestamps: np.ndarray, length: int = 168) -> tuple[int, int]:
    """Trouve une plage continue d'au plus ``length`` heures consécutives (pas de 1h)."""
    if len(timestamps) == 0:
        return 0, 0
    if len(timestamps) < 2:
        return 0, len(timestamps)

    ts_hours = pd.to_datetime(timestamps)
    diffs = np.diff(ts_hours) == pd.Timedelta(hours=1)

    best_start = 0
    best_len = 1
    cur_start = 0
    cur_len = 1

    for idx, is_step in enumerate(diffs):
        if is_step:
            cur_len += 1
            if cur_len >= length:
                return cur_start, cur_start + length
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start = idx + 1
            cur_len = 1

    return best_start, min(best_start + best_len, len(timestamps))


def plot_energy_7days(result: EnergyResult, save_path: Path) -> None:
    """Trace les observations et prédictions sur 7 jours consécutifs (168 h)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    start, end = _find_contiguous_window(result.test_timestamps, length=168)

    ts = pd.to_datetime(result.test_timestamps[start:end])
    obs = result.y_test_kw[start:end]
    pred = result.preds_kw[start:end]
    pers = result.persistence_kw[start:end]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts, obs, label="Observations réelles (kW)", color="black", linewidth=2.0)
    ax.plot(
        ts,
        pred,
        label=f"EnergyTransformer (MAE={result.transformer_metrics_kw['mae']:.3f} kW)",
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
    )
    ax.plot(
        ts,
        pers,
        label=f"Persistance t-1 (MAE={result.persistence_metrics_kw['mae']:.3f} kW)",
        color="#1f77b4",
        linestyle=":",
        alpha=0.75,
        linewidth=1.4,
    )
    ax.set_title("Consommation électrique horaire — 7 jours consécutifs (jeu de test)")
    ax.set_xlabel("Date et heure")
    ax.set_ylabel("Puissance active moyenne (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def train_and_evaluate_energy(
    bundle: EnergyDatasetBundle,
    max_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
    checkpoint_path: Path | None = None,
    plot_path: Path | None = None,
) -> EnergyResult:
    """Entraîne EnergyTransformer, conserve le meilleur val et compare aux références."""
    set_seed(seed)
    device = get_device()
    model = EnergyTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    train_losses: list[float] = []
    val_losses: list[float] = []

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        total_items = 0
        for xb, yb in bundle.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            total_items += bs

        train_loss = running_loss / max(total_items, 1)
        val_loss, _ = _evaluate_loader(model, bundle.val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_path)

    _, test_preds_norm = _evaluate_loader(model, bundle.test_loader, criterion, device)
    elapsed = time.perf_counter() - t0

    # Retour à l'échelle physique en kW
    y_test_kw = bundle.y_test.numpy().reshape(-1) * bundle.std + bundle.mean
    preds_kw = test_preds_norm * bundle.std + bundle.mean
    persistence_kw = bundle.X_test[:, -1, 0].numpy().reshape(-1) * bundle.std + bundle.mean
    daily_kw = bundle.X_test[:, 0, 0].numpy().reshape(-1) * bundle.std + bundle.mean

    result = EnergyResult(
        num_params=count_parameters(model),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        transformer_metrics_kw=compute_regression_metrics(y_test_kw, preds_kw),
        persistence_metrics_kw=compute_regression_metrics(y_test_kw, persistence_kw),
        daily_metrics_kw=compute_regression_metrics(y_test_kw, daily_kw),
        y_test_kw=y_test_kw,
        preds_kw=preds_kw,
        persistence_kw=persistence_kw,
        daily_kw=daily_kw,
        test_timestamps=bundle.test_timestamps,
        elapsed_sec=elapsed,
    )

    if plot_path is not None:
        plot_energy_7days(result, plot_path)

    return result

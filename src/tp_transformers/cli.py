"""Interface en ligne de commande pour exécuter chaque partie du TP Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

from tp_transformers.exercise1_energy import (
    download_energy_dataset,
    generate_realistic_energy_fallback,
    prepare_energy_data,
    train_and_evaluate_energy,
)
from tp_transformers.exercise2_minigpt import train_minigpt
from tp_transformers.exercise3_minibert import train_minibert
from tp_transformers.part1_sequences import run_part1_sequence_benchmark


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_part1_seq(root: Path) -> None:
    print("=== Première Partie : RNN vs LSTM vs MiniTransformer (y = sin(x)) ===")
    _, results = run_part1_sequence_benchmark(
        save_plot_path=root / "artifacts" / "part1_sequences.png"
    )
    for r in results:
        print(
            f"{r.name:16s} | params={r.num_params:4d} | "
            f"MSE={r.test_mse:.6f} | RMSE={r.test_rmse:.6f} | "
            f"MAE={r.test_mae:.6f} | temps={r.elapsed_sec:.2f}s"
        )


def _run_part1_pretrained() -> None:
    from tp_transformers.part1_pretrained import BERTMaskPredictor, GPT2Generator

    print("=== Première Partie : GPT-2 (génération) & BERT (masquage) ===")
    gpt = GPT2Generator()
    gpt_res = gpt.generate()
    print(f"[GPT-2] shape={gpt_res.input_shape} -> {gpt_res.generated_text}\n")

    bert = BERTMaskPredictor()
    for prompt in [
        "Rabat est la [MASK] du Maroc.",
        "Paris est la [MASK] de la France.",
    ]:
        res = bert.predict_masked(prompt, top_k=5)
        print(f"[BERT] {prompt} (shape={res.input_shape})")
        for cand in res.candidates:
            print(f"  {cand.probability:6.2%} | {cand.completed_sentence}")


def _run_ex1(root: Path, source: str = "auto") -> None:
    print("=== Exercice 1 : EnergyTransformer (UCI Household Power) ===")
    uci_txt = root / "data" / "household_power_consumption.txt"
    fallback_txt = root / "data" / "household_power_consumption_synthetic_fallback.txt"

    if source == "uci" or (source == "auto" and uci_txt.exists() and uci_txt.stat().st_size > 0):
        txt_path = download_energy_dataset(root / "data")
    elif source == "fallback":
        txt_path = generate_realistic_energy_fallback(fallback_txt)
    else:
        print(
            "Dataset UCI complet absent localement : utilisation du jeu de repli "
            "réaliste au format UCI (passer --source uci pour télécharger depuis UCI)."
        )
        txt_path = generate_realistic_energy_fallback(fallback_txt)

    bundle = prepare_energy_data(txt_path)
    print(
        f"Heures={bundle.n_hours} | Train={bundle.n_train} | "
        f"Val={bundle.n_val} | Test={bundle.n_test} | "
        f"mean={bundle.mean:.4f} kW | std={bundle.std:.4f} kW"
    )
    res = train_and_evaluate_energy(
        bundle,
        checkpoint_path=root / "artifacts" / "energy_transformer.pt",
        plot_path=root / "artifacts" / "ex1_energy_7days.png",
    )
    print(
        f"Meilleure époque={res.best_epoch} (val_loss={res.best_val_loss:.4f}) | "
        f"params={res.num_params} | temps={res.elapsed_sec:.1f}s"
    )
    print(
        f"EnergyTransformer : MAE={res.transformer_metrics_kw['mae']:.4f} kW | "
        f"RMSE={res.transformer_metrics_kw['rmse']:.4f} kW"
    )
    print(
        f"Persistance (t-1) : MAE={res.persistence_metrics_kw['mae']:.4f} kW | "
        f"RMSE={res.persistence_metrics_kw['rmse']:.4f} kW"
    )
    print(
        f"Réf. journalière  : MAE={res.daily_metrics_kw['mae']:.4f} kW | "
        f"RMSE={res.daily_metrics_kw['rmse']:.4f} kW"
    )


def _run_ex2(root: Path) -> None:
    print("=== Exercice 2 : Mini-GPT caractères ===")
    res = train_minigpt(
        root / "data" / "corpus.txt",
        checkpoint_path=root / "artifacts" / "minigpt.pt",
        plot_path=root / "artifacts" / "ex2_minigpt_loss.png",
    )
    print(
        f"Corpus={res.corpus_chars} chars | vocab={res.vocab_size} | "
        f"params={res.num_params} | causal_max_diff={res.causal_max_diff:.2e} | "
        f"best_val_loss={res.best_val_loss:.4f} (ppl={res.val_perplexity:.2f}) | "
        f"temps={res.elapsed_sec:.1f}s"
    )
    for temp, txt in res.generations.items():
        print(f"\n--- Température {temp} ---\n{txt}")


def _run_ex3(root: Path) -> None:
    print("=== Exercice 3 : Mini-BERT mots (MLM) ===")
    res = train_minibert(
        root / "data" / "bert_corpus.txt",
        checkpoint_path=root / "artifacts" / "minibert.pt",
        plot_path=root / "artifacts" / "ex3_minibert_training.png",
    )
    print(
        f"Phrases train={res.n_train} val={res.n_val} | vocab={res.vocab_size} | "
        f"params={res.num_params} | padding_max_diff={res.padding_max_diff:.2e} | "
        f"val_acc={res.best_val_accuracy:.2%} | temps={res.elapsed_sec:.1f}s"
    )
    for pred in res.sample_predictions:
        top3 = ", ".join(f"{tok} ({p:.1%})" for tok, p in pred.candidates)
        print(f"  {pred.sentence} -> {top3}")


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur CLI pour tp-transformers."""
    parser = argparse.ArgumentParser(
        prog="tp-transformers",
        description="Exécution des modèles et exercices du TP Architecture Transformer.",
    )
    parser.add_argument(
        "command",
        choices=["part1-seq", "part1-pretrained", "ex1", "ex2", "ex3", "all"],
        help="Sous-commande à exécuter.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "uci", "fallback"],
        default="auto",
        help="Source des données pour l'Exercice 1 (défaut : auto).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée principal de la CLI."""
    args = build_parser().parse_args(argv)
    root = _project_root()

    if args.command == "part1-seq":
        _run_part1_seq(root)
    elif args.command == "part1-pretrained":
        _run_part1_pretrained()
    elif args.command == "ex1":
        _run_ex1(root, source=args.source)
    elif args.command == "ex2":
        _run_ex2(root)
    elif args.command == "ex3":
        _run_ex3(root)
    elif args.command == "all":
        _run_part1_seq(root)
        _run_part1_pretrained()
        _run_ex1(root, source=args.source)
        _run_ex2(root)
        _run_ex3(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

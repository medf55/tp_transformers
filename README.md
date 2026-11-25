# Architecture Transformer (`PyTorch + HuggingFace Transformers`)

Application modulaire couvrant l'**Activité Pratique N° 4 — Architecture Transformer** : comparaison séquentielle (`SimpleRNN`, `SimpleLSTM`, `MiniTransformer`), modèles préentraînés (`GPT-2`, `BERT`), prévision énergétique horaire (`EnergyTransformer`), `MiniGPT` causal et `MiniBERT` bidirectionnel entraînés depuis zéro.


---

## Couverture du TP

| Étape | Description | Module |
|---|---|---|
| **Partie 1 (1–4)** | Série $y=\sin(x)$ et comparaison `SimpleRNN`, `SimpleLSTM`, `MiniTransformer` | `src/tp_transformers/part1_sequences.py` |
| **Partie 1 (5–6)** | Génération autorégressive `GPT-2` et complétion `[MASK]` avec `BERT` multilingue | `src/tp_transformers/part1_pretrained.py` |
| **Exercice 1** | Prédiction horaire de consommation électrique (`EnergyTransformer` vs baselines) | `src/tp_transformers/exercise1_energy.py` |
| **Exercice 2** | `MiniGPT` caractères from scratch, masque causal et températures `0.5 / 1.0 / 1.5` | `src/tp_transformers/exercise2_minigpt.py` |
| **Exercice 3** | `MiniBERT` mots from scratch (`max_len=32`), masquage MLM 15 % et top-3 | `src/tp_transformers/exercise3_minibert.py` |
| **Socle & CLI** | Graines aléatoires, métriques, `HF_HUB_DISABLE_XET=1`, CLI `tp-transformers` | `src/tp_transformers/utils.py`, `cli.py` |
| **Tests & Notebook** | Suite de 13 tests `pytest` et notebook pédagogique exécuté de bout en bout | `tests/`, `notebooks/` |

---

## Points clés d'ingénierie IA

- **Prévention stricte du *Data Leakage*** : découpages chronologiques (`150/40` sur $\sin(x)$, `70 % / 15 % / 15 %` en énergie), normalisation (`mean = 0.9523 kW`, `std = 0.4477 kW`) et vocabulaire `WordTokenizer` calculés exclusivement sur l'entraînement.
- **Invariants architecturaux vérifiés en `eval()`** : écart maximal nul (`0.00e+00`) sur les logits passés lors de la modification du futur dans `MiniGPT` (masque causal triangulaire), et `0.00e+00` sur les tokens utiles lors de la modification des positions `PAD` dans `MiniBERT` (`enable_nested_tensor=False`).
- **Performance et convergence mesurées** : sur $y=\sin(x)$, `SimpleRNN` (`0.000028`), `SimpleLSTM` (`0.000003`) et `MiniTransformer` (`0.000006` MSE test) ; sur l'Exercice 1 (jeu de repli horaire 8 760 h au format UCI), `EnergyTransformer` atteint `0.1219 kW` MAE (`0.1531 kW` RMSE) devant la Persistance (`0.1717 kW`) et la Référence journalière (`0.2326 kW`) ; `MiniBERT` atteint `84.97 %` d'exactitude MLM en validation.
- **Robustesse hors-ligne & Windows** : `HF_HUB_DISABLE_XET=1` configuré avant l'import `transformers`, mode de repli déterministe pour `GPT-2` / `BERT` (`local_files_only`) et générateur réaliste au format UCI lorsque l'archive externe (`20,6 Mo`) ou les poids HuggingFace (`1,26 Go`) ne sont pas présents localement.

---

## Démarrage rapide

### 1. Prérequis et installation

```powershell
# Créer l'environnement Python 3.12 et installer les dépendances
uv sync
```

### 2. Commandes CLI (`tp-transformers`)

```powershell
# Afficher l'aide et les options disponibles
uv run tp-transformers --help

# Première Partie : benchmark RNN vs LSTM vs MiniTransformer
uv run tp-transformers part1-seq

# Première Partie : GPT-2 et BERT complets (téléchargement ~1,26 Go au premier lancement)
uv run tp-transformers part1-pretrained

# Exercices 1, 2 et 3 (--source uci pour télécharger le jeu UCI officiel sur Ex. 1)
uv run tp-transformers ex1
uv run tp-transformers ex2
uv run tp-transformers ex3
```

### 3. Tests et Notebook

```powershell
# Suite de tests unitaires et d'invariants (12 passed, 1 live test skipped par défaut)
uv run pytest -v

# Activer également le test live HuggingFace (gpt2 + bert-base-multilingual-cased)
$env:TP_RUN_LIVE_HF="1"; uv run pytest -v
```

- Notebook complet avec sorties exécutées et réponses aux questions : [`notebooks/tp_transformers_solution.ipynb`](./notebooks/tp_transformers_solution.ipynb)

# Agents

This repository predicts in-hospital death for the ERAMIA-RS 2026 code challenge. Human-facing docs are Brazilian Portuguese. This file and `docs/agents/` are English.

The public training CSV is not in git. Put `final_treino_val.csv` at `data/final_treino_val.csv`. Do not commit it, and do not commit the holdout `teste.csv`. See `data/README.md`.

## Commands

```bash
uv sync
uv run desfecho train --data data/final_treino_val.csv --out artifacts
uv run desfecho evaluate
uv run desfecho visualize --data data/final_treino_val.csv
uv run desfecho predict --data teste.csv --out submissao.csv
uv run desfecho package --out dist/eramia-rs-2026.zip
uv run pytest
```

`predict` must stay offline: no downloads, no refit, no threshold search. The threshold lives in `artifacts/modelo.joblib` and is copied in `artifacts/limiar.json`. Class 1 means the six-decimal probability written to the CSV is `>=` that threshold.

## TabPFN

An isolated experiment lives in `experiments/tabpfn`. It is not part of `uv sync` at the repo root and must not replace `artifacts/modelo.joblib`. Read `docs/agents/tabpfn-experiment.md`. The manifest stays `not_run` until a machine actually finishes the fits. Do not commit weights, the training CSV, or per-patient scores.

## Do not

- Report a metric that is not in `artifacts/metrics.json` or that you just recomputed from `artifacts/oof.csv`. The README numbers are the committed run (seed 42, 5×5).
- Fit encoders, splines, imputers, or early stopping on rows whose labels you are scoring.
- Use `ID` as a feature. A numeric-ID AUC near 0.5 is a sanity check only.
- Tune hyperparameters on the same CV you quote. The fixed settings are in `src/desfecho/config.py`.
- Turn on class weights, SMOTE, or isotonic calibration. The operating point is the threshold, not the loss.
- Add TabPFN or any weight that has to be downloaded at predict time. CPU-only logistic regression + CatBoost is the shipped path.
- Claim the columns are SIVEP fields. That mapping is an inference and several parts of it do not fit this file. See `docs/nota-de-pesquisa.md`.
- Put secrets or tokens in the repo. Keep commit messages plain.

## Layout

`src/desfecho/data.py` loads and checks. `models.py` is the spline logistic and CatBoost. `cv.py` runs repeated stratified CV and averages fold models at predict time. `threshold.py` picks the single cut. `metrics.py` is AUC, LR−, F1, calibration, DeLong, bootstrap. `cli.py` is Typer. `store.py` writes the artifact directory.

Portuguese teaching note: `docs/abordagem.md`. Research note with links: `docs/nota-de-pesquisa.md`. Zip blurb: `docs/ferramentas.md`.

## Commits

Small, ordered, in Portuguese, as the repo author. The training file stays ignored.

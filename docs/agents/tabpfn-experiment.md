# TabPFN v2 experiment

Preparation only. No Mac run, no MPS measurement, no AUC gain. `experiments/tabpfn/manifest.json` says `not_run`. Do not edit `artifacts/modelo.joblib` or the root dependency list.

## Frozen comparison

`desfecho.cv.official_splits` is the 5×5 used by training (`TrainConfig` seed 42). The validation-index sha256 for the 6,332-row file is in the manifest. `recommend` reads only the patient-level paired AUC delta of TabPFN v2 minus the shipped ensemble. It does not read fold AUCs, the 0.010 tie band, the three-model mean, or LR−/F1.

Protocol version 2, set before any TabPFN scores exist. Holding out a repeated-CV repeat does not hold out patients. `audit_threshold_patients` shows the label intersection and that the models which scored the selection patients were trained on the evaluation group. A label-disjoint group cut is still exploratory for that reason. True nesting would be 5 outer groups times 25 inner fits (125 TabPFN fits) and is not scheduled. `recommend` never returns `adopt`. A favorable AUC interval stays `inconclusive` with reason `auc_favoravel_limiar_apenas_exploratorio`.

Uncertainty:

- Patient bootstrap and DeLong condition on the stored scores. They do not refit models and do not redo model selection. They are not a sample of 25 folds.
- The five repeat-wise AUC deltas are dependent. Report min, mean, and max. Do not turn them into a t interval.
- If the two intervals disagree about covering zero, the decision is inconclusive.
- The shipped ensemble was selected on this same out-of-fold table. The contrast is against the delivered model. That optimism is labeled, not corrected.

A fresh baseline refit must match `artifacts/oof.csv` column `p_ensemble` within 1e-4 in ID order. Otherwise stop with `baseline_nao_reproduzida`.

## Versions and licenses

Checked from the PyPI metadata, the 9.2.0 wheel, and the Hugging Face model card. Not from a Mac run.

- Code package `tabpfn==9.2.0`, license Apache-2.0. Wheel sha256 is in the manifest.
- Package default `model_version` is v3.5. v3.5 calls `ensure_license_accepted`. Do not construct `TabPFNClassifier()` with defaults.
- Weights: `Prior-Labs/TabPFN-v2-clf`, file `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, 29009539 bytes, sha256 in the manifest. Card license is `priorlabs-1-1` (Apache 2.0 plus attribution paragraph 10). Not gated. The code's S3 fallback URL returned 404 on 2026-11-05.
- v2 CPU guard is 1000 rows (`cpu_sample_limit`). A training fold is about 5066 rows, under the v2 pretraining cap of 10000 samples and 500 features. CPU full fits set `settings.tabpfn.allow_cpu_large_dataset` and record the override. They do not set `ignore_pretraining_limits`.
- The library contains MPS paths. Availability on this Mac is untested. `select_device` runs a real tiny fit on CPU, then on MPS only if `torch.backends.mps.is_available()`. MPS is selected only when that fit returns finite probabilities.

`n_estimators` is pinned to 8.

## Commands

The system Python is 3.15.2 and must not be the venv. `experiments/tabpfn/.python-version` is 3.12 and `requires-python` is `>=3.12,<3.13`. `uv sync` downloads that interpreter into uv's own directory. It does not replace Homebrew Python.

Install peak used by the gate is 2 GiB. That is four times the macOS arm64 wheel sum in the lock (about 350 MB, torch 2.14.1 cp312 macosx_14_0_arm64 is 127 MB) plus a managed CPython. It is not a measured install on the Mac. The gate also keeps 5 GiB free. A full run also wants at least 6 GiB available on a machine with at least 15 GiB of RAM. Set `UV_CACHE_DIR` to `experiments/tabpfn/.uv-cache` so the download does not use the global cache.

From the unpacked root, after `python3 experiments/tabpfn/preflight.py` and `uv sync` inside `experiments/tabpfn`:

```bash
uv run tabpfn-exp fetch --dest weights
uv run --offline tabpfn-exp run \
  --weights weights/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt \
  --out runs/oficial \
  --data ../../data/final_treino_val.csv \
  --baseline-oof ../../artifacts/oof.csv
```

The first `run` stops after smoke and writes `runs/oficial/smoke.json` plus a linear time projection labeled untested. Add `--confirm-full-run` only after reading that projection. `uv run --offline` is the uv flag and must come before the command. The runner also sets `HF_HUB_OFFLINE=1` before fit.

`fetch` takes no token. Patient CSV, checkpoints, and per-patient arrays stay under `weights/` and `runs/`, which are gitignored.

Handoff zip, from the repo root, contains protocol and code only:

```bash
uv run --project experiments/tabpfn tabpfn-exp handoff --root . --out dist/tabpfn-exp-handoff.zip
```

## Tests that ran here

`tests/test_tabpfn_exp.py` checks split coverage, the decision gates, CPU fallback when an MPS smoke fails, refusal to invent a runtime without a smoke, sanitizing, and fold resume. It does not fit TabPFN.

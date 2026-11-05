# Architecture

English notes for code changes. The clinical write-up is `docs/abordagem.md`.

## Data contract

`src/desfecho/schema.py` is the column list. Predictors are 23 binaries, `sociodemografico3` (age), and two nominals (`sociodemografico4`, `sociodemografico5`). `ID` and `DESFECHO` never enter the matrix. `prepare_frame` coerces binaries and age to float and nominals to stable strings (`26.0` and `26` both become `"26"`). Missing stays missing.

Column names are matched case-insensitively (`ID`/`id`, `DESFECHO`/`desfecho`). The submission header is lowercase `id,prob,desfecho`, as the rules require.

## Models

`LogisticDesign` fits, on the training rows only:

- median imputation for numerics (unused on the official file, which has no gaps);
- a B-spline on age (`SplineTransformer`, 5 quantile knots, degree 3, `include_bias=False`, linear extrapolation past the boundary knots);
- one-hot nominals with the training-fold mode as the dropped reference. Unseen levels collapse to that reference.

`CatBoostFrameModel` lists the two nominals as `cat_features`. Early stopping uses a stratified 20% slice of the outer training fold, then a refit on the full outer training fold with `best_iteration + 1` trees. The validation fold is not in that slice.

`TrainConfig` in `config.py` holds the pre-specified hyperparameters. Change them only if you also stop quoting the old `metrics.json`.

## Cross-validation and the file that ships

`run_training` uses `RepeatedStratifiedKFold(5, 5, random_state=seed)`. Every model sees the same indices. Out-of-fold predictions are stored per repeat, then averaged per patient. That average is the score for AUC, calibration, and the threshold.

The shipped score is chosen by `_choose_shipped`:

- equal-weight average of logistic and CatBoost by default;
- the better of those two if their mean-OOF AUCs differ by more than 0.02;
- age-only logistic if the ensemble is more than 0.005 below it.

`predict_proba` averages the saved fold models of the shipped family. It does not look at labels and does not recompute the threshold.

## Threshold

`select_threshold` minimizes the bootstrap 75th percentile of LR− on a quantile grid, subject to at least 8% predicted zeros and F1 within 0.08 of the best F1 on that grid. Specificity zero is rejected (LR− is infinite in the rules; the bootstrap uses `1e6` as a stand-in so a fragile cut looks bad). Nested repeats in `cv.py` re-select without the evaluated repeat. The number stored in the bundle is the one fit on all repeats.

## Submission

`build_submission` rounds probabilities to 6 decimals, parses them back, and sets the label from the parsed value. `validate_submission` checks the header, range, unique ids, order, and exact agreement with the threshold.

## Artifacts

`modelo.joblib` is the bundle (models, threshold, config, which family shipped). `oof.csv` is one row per training patient. `metrics.json` includes ROC points, so it is larger than the summary table. Figures are redrawn by `visualize` from those files; they are not a second fit.

## What is intentionally absent

No TabPFN import, no imbalanced-learn, no isotonic regression, no target encoding, no feature selection step. Adding any of those is a modeling change and belongs in `docs/nota-de-pesquisa.md` with a reason, not only in a pull request diff.

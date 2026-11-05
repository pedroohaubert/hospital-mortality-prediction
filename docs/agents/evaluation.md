# Evaluation

Do not invent or round beyond what the code prints. `desfecho evaluate` formats the committed `artifacts/metrics.json`.

## What the headline AUC is

For each patient, average the out-of-fold probabilities across the five repeats. Compute ROC AUC once on those 6,332 scores. That is `modelos.<name>.auc_media_oof`.

The bootstrap interval resamples patients, stratified by label, with the scores held fixed (2,000 draws, percentile 2.5 and 97.5). The DeLong interval is the large-sample variance of that same AUC (`metrics.delong_auc`). They are not a confidence interval for the unseen 1,605-row test, and they are not the standard deviation of the five repeat AUCs. That standard deviation is about 0.001 because each repeat scores the same people. Quote `auc_repeticoes_dp` only if you say that.

Paired model comparisons use DeLong on the two mean-OOF score vectors (`comparacoes_delong`). A delta whose interval covers zero is not a win.

## Operating point

`ponto_oof` is sensitivity, specificity, LR−, and F1 at the shipped threshold on the same mean-OOF scores used to pick it. It is optimistic. Say so, matching `ponto_oof_aviso`.

`ponto_aninhado_media` is a bad name. Each repeat scores every patient, so choosing the threshold on the other repeats still uses the labels of the patients being scored. It is not patient-disjoint and it is not nested cross-validation. The shipped JSON keeps the old key. Treat that LR− and F1 as exploratory, same as `ponto_oof`. The single threshold in the bundle is still the one the submission uses. The patient bootstrap and DeLong intervals condition on the stored scores. They do not include refitting or model selection.

LR− is `(1 - sensitivity) / specificity`. Specificity 0 must surface as non-finite, never as zero. F1 is `2 TP / (2 TP + FP + FN)`.

Calibration slope and intercept are a logistic regression of the label on the logit of the clipped probability, with a very large `C`. Slope near 1 and intercept near 0 means the probabilities already match the outcome rate. Brier and quantile ECE are next to them.

## Leakage checks that should stay green

- `ID` is not in `LogisticDesign.get_feature_names_out`.
- Spline medians and category levels are fit on the training index only (`tests/test_train.py`).
- Early stopping indices are a split of the outer training fold.
- `predict` drops a present `DESFECHO` and warns.
- The submission label is recomputed from the rounded probability and the stored threshold.

## Tests

`uv run pytest` uses synthetic frames, not the challenge CSV. The integration test runs a 2×2 CV with a few dozen trees. It does not reproduce the official AUC. After a real `train`, trust `artifacts/metrics.json`, not the synthetic run.

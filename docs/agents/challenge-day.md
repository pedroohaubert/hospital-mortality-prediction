# Challenge day

The holdout file arrives as `teste.csv` (1,605 rows in the rules), with the same predictors and without labels. The team runs its own code offline.

Before the network is gone:

```bash
uv sync
uv run desfecho train --data data/final_treino_val.csv
```

The committed `artifacts/modelo.joblib` is already a trained bundle. Re-training is only required if the code or the training file changed. Check `artifacts/limiar.json` afterwards so the threshold you think you are using is the one in the file.

On the day:

```bash
uv run --offline desfecho predict --data teste.csv --model artifacts/modelo.joblib --out submissao.csv
```

That command:

- does not open a socket and does not download weights;
- does not refit;
- does not search for a new threshold;
- ignores `DESFECHO` if the file still has it;
- writes `id,prob,desfecho` in the input row order;
- refuses to write a file whose labels disagree with `prob >= threshold`.

If `uv run --offline` cannot see the environment, the sync step was skipped. Do not improvise a `pip install` against the network. Fix the environment before the round starts.

Quick checks on the CSV before handing it in:

- 1,605 data rows plus the header, unless the organizers actually shipped a different count;
- no empty `prob` or `desfecho`;
- `prob` between 0 and 1;
- ids unique and identical to `teste.csv`;
- a non-trivial share of zeros (the training operating point was about 8% predicted recovery). If every row is 1, specificity is zero and LR− is infinite. Do not "fix" that by editing labels by hand. That breaks the single-threshold rule and is a disqualification.

The zip for the jury:

```bash
uv run desfecho package --out dist/eramia-rs-2026.zip
```

It contains the source, `uv.lock`, the model, the threshold, and `docs/ferramentas.md`. It does not contain the training CSV.

Do not peek at any label that shows up in the test file, and do not re-cut the threshold after seeing the test distribution.

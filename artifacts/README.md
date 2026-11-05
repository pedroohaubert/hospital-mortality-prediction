# Artefatos da validação

Gerados por `desfecho train` na base de 6.332 linhas, com semente 42, 5 folds e 5 repetições.

- `modelo.joblib` — modelos de cada fold e o limiar. É o que o `predict` carrega.
- `limiar.json` — o mesmo corte, legível, mais o ponto de operação.
- `metrics.json` — AUC, intervalos, calibração, comparações e a curva do limiar.
- `oof.csv` — uma linha por paciente do treino: `id`, `desfecho` e as probabilidades médias out-of-fold. Não traz as features. Serve para recomputar a AUC sem retreinar.
- `explicacao.json` — importância do CatBoost, coeficientes das binárias e a curva de idade.

O CSV original do desafio não está aqui. Ver `data/README.md`.

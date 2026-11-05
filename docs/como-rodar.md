# Como rodar

Precisa do Python 3.10 ou mais novo e do [uv](https://docs.astral.sh/uv/). O lockfile está na raiz. Eu treinei com Python 3.12.

## Treino

O arquivo `final_treino_val.csv` fica em `data/` e não vai para o git. São 6.332 linhas, com `ID`, as preditoras e `DESFECHO`.

```bash
uv sync
uv run desfecho train --data data/final_treino_val.csv --out artifacts
```

Isso faz a validação 5×5, escolhe o limiar, grava `artifacts/modelo.joblib` e escreve as métricas. Na máquina em que medi, o comando levou pouco menos de um minuto, em CPU, sem GPU.

Para olhar o resultado:

```bash
uv run desfecho evaluate
uv run desfecho visualize --data data/final_treino_val.csv --out figs
```

`evaluate` só lê o que já foi gravado. `visualize` redesenha as figuras a partir do `oof.csv` e do `metrics.json`.

## Predição

No dia da prova o arquivo é o `teste.csv`, com os mesmos preditores e sem o rótulo. Um comando:

```bash
uv run --offline desfecho predict --data teste.csv --model artifacts/modelo.joblib --out submissao.csv
```

O `--offline` evita que o uv tente a rede. O ambiente precisa ter sido instalado antes, com `uv sync`, enquanto ainda havia internet. O `predict` em si não baixa modelo nenhum: os folds já estão no `joblib`.

A saída tem o cabeçalho `id,prob,desfecho`. A probabilidade é gravada com seis casas. A classe é 1 se essa probabilidade, já arredondada, for maior ou igual ao limiar guardado no modelo. Se as duas coisas não baterem, o comando falha em vez de entregar um CSV inconsistente.

## Testes

```bash
uv run pytest
```

Os testes usam dados sintéticos. Não dependem do CSV do desafio.

## Notebooks

No Colab, abra `notebooks/01_exploracao.ipynb` ou `notebooks/02_modelo.ipynb`. O primeiro só explora o CSV. O segundo treina de novo e mostra o limiar. Se o arquivo não estiver no repositório, a primeira célula pede o upload.

Fora do Colab, com o uv já sincronizado:

```bash
uv run python -m ipykernel install --user --name desfecho
```

## Zip da submissão

```bash
uv run desfecho package --out dist/eramia-rs-2026.zip
```

Entram o código, o lockfile, o modelo, o limiar e o `docs/ferramentas.md`. O CSV de treino fica de fora.

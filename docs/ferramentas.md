# Linguagens, bibliotecas e ferramentas

Documento para acompanhar o código no zip da ERAMIA-RS 2026.

## Linguagem

Python 3.12. O pacote declara compatibilidade com 3.10 ou mais novo. Não há etapa em R, SQL ou outra linguagem.

## Ambiente

- [uv](https://docs.astral.sh/uv/) para criar o ambiente e travar as versões (`uv.lock`)
- CLI com Typer e Rich: comandos `train`, `evaluate`, `predict`, `visualize` e `package`
- joblib para gravar os modelos dos folds
- Jupyter e Google Colab para os notebooks em `notebooks/`
- pytest para os testes

Não usei GPU. O CatBoost e a regressão logística rodam em CPU. Não há peso baixado de serviço externo.

## Bibliotecas no treino que gerou o modelo

Versões lidas do ambiente desse treino, também gravadas em `artifacts/metrics.json`:

| Biblioteca | Versão | Para quê |
| --- | --- | --- |
| numpy | 2.5.4 | arrays e reamostragem |
| pandas | 3.1.1 | leitura do CSV |
| scikit-learn | 1.9.1 | logística, spline, folds, AUC, Brier |
| scipy | 1.18.1 | quantis e a distribuição normal do teste de DeLong |
| catboost | 1.2.10 | gradient boosting com categóricas |
| matplotlib | 3.11.2 | figuras |
| typer | 0.27.3 | CLI |
| rich | 15.0.0 | tabelas no terminal |
| joblib | 1.6.0 | persistência do modelo |

## O que o `predict` faz

Lê `teste.csv`, aplica a média dos modelos de fold já treinados, compara com o limiar único gravado em `artifacts/modelo.joblib` e escreve `submissao.csv`. Não acessa a rede e não reescolhe o limiar.

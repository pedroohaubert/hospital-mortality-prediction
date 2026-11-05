# Dados

Coloque aqui o arquivo `final_treino_val.csv` do Code Challenge, com esse nome.

Ele não entra no git. O regulamento não autoriza republicar a base: os atributos foram renomeados para dificultar a identificação da fonte, e a organização só revela essa fonte depois das submissões. O arquivo que eu usei localmente é o de treino e validação, com 6.332 linhas.

No dia da competição, o `teste.csv` (1.605 linhas, sem o rótulo) fica fora desta pasta também. A previsão aponta para o caminho que você passar no comando.

```bash
uv run desfecho train --data data/final_treino_val.csv
uv run desfecho predict --data teste.csv --out submissao.csv
```

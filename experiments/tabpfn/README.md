# Experimento TabPFN v2

Built with PriorLabs-TabPFN.

Isto ainda não foi rodado no Mac. O manifest diz `not_run`. Não há ganho medido, e não há prova de que o MPS desta máquina funciona.

O modelo entregue continua o ensemble de logística e CatBoost. Este diretório não troca o `artifacts/modelo.joblib` e não entra no `uv sync` da raiz.

## O que está congelado

A pergunta é se o TabPFN **v2** melhora o ensemble já entregue, no mesmo 5×5, com peso local e sem rede na hora de prever. A regra está em `PROTOCOLO.md`. Ela não adota sozinha. LR− e F1 são exploração, porque o corte por repetição reusa o rótulo dos mesmos pacientes. A faixa de 0,010 e o melhor fold não entram.

## O que eu conferi no pacote, sem rodar o modelo

O pacote no PyPI é o `tabpfn` 9.2.0. A licença do código é Apache-2.0. O default interno desse pacote é o modelo **v3.5**, que passa por aceite de licença no navegador. Este experimento não usa esse default. Ele chama `create_default_for_version(ModelVersion.V2)` e aponta para um arquivo local.

O peso v2 de classificação é público e não está marcado como gated. O repositório é `Prior-Labs/TabPFN-v2-clf`. O arquivo default é `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, com 29.009.539 bytes e sha256 `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`. A licença desse peso é a Prior Labs 1.1, não a Apache. Ela é a Apache com um parágrafo 10 de atribuição. Uso interno de teste não exige o letreiro. Se o relatório for entregue para fora, o letreiro é “Built with PriorLabs-TabPFN”. Não há cláusula de uso não comercial no texto que eu li. Isso é leitura da licença, não parecer jurídico.

O fallback em `storage.googleapis.com` devolveu 404 para esse arquivo em 2026-11-05. O download usa o Hugging Face, sem token na linha de comando. O `fetch` não abre o login da Prior Labs. Se o arquivo deixar de ser público, o comando falha e para.

No código do v2, CPU recusa mais de 1.000 linhas, salvo `TABPFN_ALLOW_CPU_LARGE_DATASET`. Um fold nosso tem cerca de 5.066 linhas, ainda abaixo do teto de pré-treino de 10.000 amostras e 500 features. Se o smoke escolher CPU, o fit cheio liga esse override e grava isso no log. Não liga `ignore_pretraining_limits`.

O pacote tem tratamento de MPS, inclusive erro de memória. Isso não mede este Mac. O smoke é que decide.

## Comando no Mac

O `python3` do Homebrew é 3.15.2. Torch 2.14.1 não tem roda para ele. O `.python-version` pede 3.12, e o `uv sync` baixa esse interpretador para a pasta do próprio uv. Isso não troca o Python do sistema.

Há cerca de 14 GiB livres e o piso é deixar 5 GiB. O pico previsto da instalação é 2 GiB. Esse 2 GiB não foi medido neste Mac. É quatro vezes a soma das wheels arm64 do lock, cerca de 350 MB, mais o CPython gerenciado. O cache do uv fica em `experiments/tabpfn/.uv-cache`, não no cache global.

```bash
mkdir -p "$HOME/Projetos"
unzip tabpfn-exp-handoff.zip -d "$HOME/Projetos/eramia-tabpfn-experiment"
cd "$HOME/Projetos/eramia-tabpfn-experiment"
/opt/homebrew/bin/python3 experiments/tabpfn/preflight.py
cd experiments/tabpfn
UV_CACHE_DIR="$PWD/.uv-cache" "$HOME/.local/bin/uv" sync
UV_CACHE_DIR="$PWD/.uv-cache" "$HOME/.local/bin/uv" run tabpfn-exp fetch --dest weights
UV_CACHE_DIR="$PWD/.uv-cache" "$HOME/.local/bin/uv" run --offline tabpfn-exp run \
  --weights weights/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt \
  --out runs/oficial \
  --data /CAMINHO/final_treino_val.csv
```

O `run` sem `--confirm-full-run` só faz o smoke. Ele para se o disco livre ficar abaixo de 5 GiB mais o pico, ou, no fit cheio, se a RAM disponível ficar abaixo de 6 GiB numa máquina de 16 GiB. Não apaga arquivo nenhum.

Esse `run` faz o smoke, grava `runs/oficial/smoke.json` e para. O JSON traz o device que de fato passou no fit minúsculo, o tempo, o pico de RSS e uma projeção linear das 25 idas. A projeção não é medição. A memória do fold cheio fica sem número.

Se o tempo couber, o mesmo comando ganha `--confirm-full-run`. Aí sim são 25 fits de TabPFN e o refit das baselines, nos mesmos folds. Dá para interromper e rodar de novo o mesmo comando com `--confirm-full-run`. Fold com `status: done` e o mesmo `config_hash` não repete.

O `oof.csv` publicado não vai no zip. Sem `--baseline-oof`, o contraste usa o refit local. Para checar contra o arquivo antigo, gere outro só na máquina.

```bash
uv run --project . desfecho train --data data/final_treino_val.csv --out /tmp/baseline-local
```

Passe `--baseline-oof /tmp/baseline-local/oof.csv`. Não suba esse arquivo.

O agregado sem paciente fica em `runs/oficial/aggregate.json`. Probabilidade por linha fica em `runs/` e não vai para o git.

## O que não fazer

Não copiar `weights/` nem `runs/` para o repositório. Não apontar o `predict` entregue para este peso. Não tratar CPU como GPU. Se o MPS falhar, o motivo fica em `reason` e o device gravado é `cpu`.

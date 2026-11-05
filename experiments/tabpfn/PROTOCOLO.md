# Protocolo congelado

Versão 1. Escrito antes de qualquer AUC de TabPFN neste conjunto. O manifest continua `not_run` até o Mac terminar o `run` com `--confirm-full-run`.

## Pergunta

O TabPFN v2, com 8 estimadores e o checkpoint default de classificação, melhora o ensemble entregue o bastante para trocar o modelo, sem perder o ponto de operação e sem depender de rede?

A média dos três modelos é relatório secundário. Ela não promove uma decisão.

## O que é igual ao treino entregue

- `RepeatedStratifiedKFold`, 5 folds, 5 repetições, semente 42. A função é `desfecho.cv.official_splits`.
- O hash dos índices de validação no CSV de 6.332 linhas é `e896ad31ab4a71dc5b3d4ae13c35cf8008f3276ef391feb8b989de646230d19c`. Se não bater, o programa para.
- Baselines são a logística só com idade, a logística com spline e o CatBoost, com o `TrainConfig` atual. A semente de cada fold é `42 + 10007 * repetição + 17 * fold`, a mesma do CatBoost entregue.
- O ensemble de referência é 0,5 logística mais 0,5 CatBoost, em cada repetição e na média do paciente.
- O TabPFN vê as mesmas colunas cruas, com as duas nominais marcadas como categóricas. A spline continua só na logística. O pré-processo interno do TabPFN é dele.

Antes do contraste, a média refeita do ensemble tem de chegar a menos de 0,0001 da coluna `p_ensemble` do `artifacts/oof.csv`, na mesma ordem de ID. Se não chegar, a decisão é inconclusiva e o motivo é `baseline_nao_reproduzida`.

## O que conta como evidência

A unidade do intervalo é o paciente, não o fold.

1. Cada paciente fica com a média das cinco previsões out-of-fold. Essa média é o escore do contraste principal.
2. O intervalo bootstrap reamostra pacientes, estratificado, 2.000 vezes, e recalcula a diferença de AUC. Os escores ficam fixos. Isso não inclui a variância de treinar de novo. As cinco previsões que foram médias são dependentes, porque os folds se sobrepõem.
3. O intervalo de DeLong na mesma diferença trata os escores como fixos. Se um intervalo cobre o zero e o outro não, a decisão é inconclusiva.
4. As cinco diferenças de AUC, uma por repetição, são um mínimo, uma média e um máximo. Não viram intervalo. Não se divide o desvio por raiz de cinco. Não se empilha os 25 folds como se fossem amostras independentes.

A escolha antiga do ensemble entregue olhou este mesmo out-of-fold. O contraste é contra o modelo que foi entregue, não contra um modelo travado sem ver estes rótulos. Isso pode deixar o TabPFN um pouco mais difícil de vencer. Não há correção numérica. Há o rótulo.

## Limiar

Correção da versão 2, antes de existir escore de TabPFN. Segurar uma repetição não separa pacientes. Cada repetição pontua os 6.332. O corte “da outra repetição” usa o rótulo de quem depois é medido. Isso não é validação aninhada.

Dá para escolher o corte só com os rótulos de fora de um grupo. Os folds da primeira repetição formam esses grupos. Aí o rótulo do grupo não entra na escolha. O escore de quem fica de fora, porém, veio de modelo que treinou nesse grupo. Tirar essa influência exige treinar de novo sem o grupo. Com 5 grupos externos e os 25 fits internos, seriam 125 fits de TabPFN, fora do orçamento desta rodada.

LR− e F1 ficam no agregado com `papel: exploratorio` e `entra_na_decisao: false`. Não há adoção automática. AUC favorável com checksum ok continua inconclusiva, motivo `auc_favoravel_limiar_apenas_exploratorio`. Rejeição automática só ocorre se o peso falha ou se os dois intervalos de AUC ficam abaixo de zero.

## Decisão

`recommend` só vê o contraste de AUC, TabPFN menos ensemble. As folgas de 0,02 em LR− e F1 ficaram no código como descrição antiga e não abrem adoção.

- Inconclusivo se ainda não mediu, se falta intervalo, ou se bootstrap e DeLong discordam sobre o zero.
- Rejeita se o peso não confere ou a inferência não ficou offline.
- Rejeita se os dois intervalos ficam inteiros abaixo de zero.
- Inconclusivo se os intervalos cobrem o zero.
- Inconclusivo também se os dois ficam acima de zero. O limiar que decidiria a LR− não é disjunto.

Bootstrap e DeLong reamostram pacientes com o escore já pronto. Não reamostram o treino. A faixa de 0,010 do ranking não é argumento. A AUC de um fold também não.

## Custo e device

O smoke de 48 linhas roda em CPU. Se o PyTorch disser que o MPS existe, o mesmo smoke roda em MPS. MPS só é o device do fit cheio se esse smoke devolver probabilidade finita. Falha de MPS grava o erro e segue em CPU. CPU não é chamada de GPU.

A projeção de tempo só aparece depois do smoke, com a hipótese não testada de que o tempo cresce linearmente com as linhas de treino e com os 25 fits. A memória do fold cheio não é projetada. Só o pico do smoke é número medido.

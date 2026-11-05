# Nota de pesquisa

Por que o modelo é uma logística com spline de idade mais um CatBoost, com limiar baixo e sem TabPFN. Escrevi isto para eu mesmo não esquecer o que estava decidido antes de olhar o número, e o que só deu para saber depois de medir.

A leitura da literatura e do regulamento que usei como ponto de partida foi feita em 05/11/2026. Os links abaixo são os que essa leitura abriu. Eu li o PDF do regulamento e o dicionário da ficha do SIVEP-Gripe datado de 25/05/2023 no próprio arquivo. Não baixei de novo cada paper nesta implementação. Onde a página não abriu naquela leitura, eu digo.

## O que o regulamento prende

Página do desafio: [ERAMIA-RS 2026](https://eramia-rs.sbc.org.br/2026/#/challenges). PDF: [regulamento](https://drive.google.com/file/d/19LDMbdCyp0WZh8ZMib8YCoLi7hRJzS4V/view).

Confirmei no PDF, não só no resumo:

- Classificação binária. `desfecho` 0 = cura, 1 = óbito, e 1 é a classe positiva.
- Treino e validação com 6.332 casos. Teste com 1.605, só no momento da competição.
- Os atributos foram renomeados para dificultar a identificação da fonte. A fonte e o detalhe do problema saem depois das submissões. Os dados vêm de fonte pública e não têm identificador direto de pessoa.
- Preditoras: `dadoclin1–14` binárias, `sociodemografico1–2` binárias, `sociodemografico3` quantitativa discreta, `sociodemografico4–5` nominais, `comorbidade1–7` binárias.
- Pré-processamento, engenharia e balanceamento são permitidos. O limiar faz parte do modelo e é escolha da equipe. Nada disso pode usar o rótulo verdadeiro do teste.
- Submissão: `id,prob,desfecho`. `prob` em [0, 1], sem ausente. Classe 1 se e só se `prob ≥ limiar`. Um limiar só. Inconsistência desclassifica.
- Junto vai um zip com o código e um documento das linguagens, bibliotecas e ferramentas. É o `docs/ferramentas.md`.
- Ranking: AUC arredondada a três casas, faixas de empate de 0,010 a partir da melhor; depois LR− = (1 − sensibilidade) / especificidade, infinita se a especificidade for 0; depois F1 = 2·VP / (2·VP + FP + FN). A justificativa da LR− no texto é que o falso negativo é o erro mais grave.
- A equipe vencedora é escolhida por banca a partir do desempenho na etapa final. Dúvidas: codechallenge.eramia26@gmail.com.

A edição de 2025 era outro problema, de desfecho de oportunidade de venda com texto e tabela ([PUCRS](https://portal.pucrs.br/noticias/ensino/grupo-de-processamento-de-linguagem-natural-pln-da-pucrs-apresenta-quatro-trabalhos-no-eramia-2025/)). Não achei regulamento nem base públicos de 2025, e nada indica que o desenho se repita. Não usei aquilo como modelo.

A pesquisa de 05/11 anotou que a inscrição fechava em 06/11/2026 às 23:59 e que a final é de 9 a 11/11, com o teste rodado pela própria equipe, sem depender de internet. Eu não reabri a página ao escrever este arquivo. O pipeline foi feito para esse cenário mesmo assim: `predict` é um comando, determinístico, com dependência travada e com o modelo já no disco.

## O que este arquivo de treino é, de fato

Medido em `final_treino_val.csv`, 6.332 linhas, sem ausente, sem `ID` repetido.

| Fato | Valor |
| --- | --- |
| Óbitos | 3.624 (57,2%) |
| Idade | 0 a 102, mediana 58. Abaixo de 18 anos a mortalidade é 0,286; de 80 em diante, 0,826 |
| `sociodemografico4` | 27 níveis, códigos 1 a 27. O nível 26 tem 2.122 linhas. Não é o conjunto {23, 24, 25, 26} |
| `sociodemografico5` | níveis 1 a 5. O mais comum é o 1 (3.708), depois o 4 (2.211). O 5 tem 26 linhas |
| `sociodemografico2` | só 80 positivos |
| Binárias | só 0 e 1. Não sobrou código 9 nem branco |
| AUC do `ID` numérico | 0,508 |

A prevalência de 57% é mais alta do que o prior de 30–40% que a leitura do Nordeste no Ranzani sugeria. Ou a amostra é mais grave, ou a fonte não é aquela, ou as duas coisas. O F1 sobe com a prevalência; a LR− não. Por isso a escolha do limiar não pode ser copiada de um dataset com outra base rate, mas a LR− ainda dá para comparar.

A idade domina. A AUC da logística só com spline de idade, na validação, é 0,717. `comorbidade5` é a binária mais comum entre as que se associam a óbito (2.493 positivos, média 0,50 nos óbitos e 0,25 nas curas). Várias clínicas têm média maior nas curas do que nos óbitos. O sinal entra com o sinal que tem. Eu não descartei coluna por isso.

Não fiz seleção de variáveis. Olhar a tabela inteira e depois reportar a validação da seleção seria o vazamento chato, o de [Kapoor e Narayanan, *Patterns* 2023](https://doi.org/10.1016/j.patter.2023.100804).

## A hipótese SIVEP, e onde ela quebra

A organização ainda não disse a fonte. O que segue é inferência.

O dicionário que eu li é o da ficha de SRAG hospitalizado do SIVEP-Gripe, com data 25/05/2023 no PDF. A leitura de 05/11 usou o [dicionário 2019–2025](https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/dicionario-de-dados-2019-a-2025.pdf) ligado no [OpenDataSUS, SRAG 2019 a 2026](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026). Não afirmo que os dois PDFs sejam o mesmo arquivo. No que eu li, `EVOLUCAO` é 1 cura, 2 óbito, 3 óbito por outras causas, 9 ignorado. Há 13 sintomas no campo 34 (febre, tosse, dor de garganta, dispneia, desconforto respiratório, saturação, diarreia, vômito, dor abdominal, fadiga, perda de olfato, perda de paladar, outro) e um bloco maior de fatores de risco no campo 35, com codificação 1 / 2 / 9. O desafio tem 14 clínicas e 7 comorbidades. Alguma coluna a mais entrou nas clínicas, e várias comorbidades ficaram de fora. Quais, eu não sei.

O que a hipótese acerta: binárias de sintoma e comorbidade, uma idade, nominais no lugar de UF e de raça ou escolaridade, desfecho cura versus óbito. O que ela não acerta neste arquivo:

- `sociodemografico4` não está restrito a 23–26. São 27 códigos, de 1 a 27. Vinte e sete é o número de UFs, então ainda pode ser UF recodificada, mas o código IBGE de UF não é 1–27, e o nível modal 26 não me diz qual estado é. Tratei como nominal, sem ordem.
- A heurística “se o 4 dominar `sociodemografico5`, é raça/cor parda” não dispara. O nível modal é o 1, não o 4. Pode ser raça com outra agregação, escolaridade ou outra coisa. Continua nominal.
- Não há 9 nem branco. Se a origem era SIVEP, alguém já recodificou o ignorado ou jogou o registro fora. Não tenho como recuperar “não preenchido”, que às vezes carrega informação de qualidade do hospital. Não imputo o que não está faltando.
- A idade cabe em anos (0–102). Não vi bebê codificado em dias. Não converti unidade.
- A prevalência não reproduz o 38% do [Ranzani et al., *Lancet Respir Med* 2021](https://doi.org/10.1016/S2213-2600(20)30560-9) nem o contraste Nordeste/Sul citado de lá (31% contra 15% abaixo de 60 anos). Uso o Ranzani como descrição de mortalidade hospitalar no SIVEP, não como prova de que esta amostra é aquela.

Por isso o README não diz que o modelo é de SRAG. Diz que a fonte é desconhecida e que a ficha do SIVEP é a hipótese mais próxima que eu encontrei.

O `ID` tem o formato do exemplo do regulamento (`A` e seis dígitos). A AUC dele é 0,508. Fica de fora. Se no teste a ordem for temporal, usar o `ID` do treino seria decorar uma sequência que não se repete.

## Literatura que justifica a faixa, não o meu número

Meu número medido é AUC 0,737 na média out-of-fold do ensemble (IC bootstrap 0,725–0,750; IC de DeLong 0,725–0,750). A tabela abaixo é o que outros trabalhos reportaram com dados parecidos. Não é meta, e não é o que eu “deveria” ter tirado.

| Trabalho | O que usaram | O que reportaram |
| --- | --- | --- |
| [Baqui et al., *Sci Rep* 2021](https://doi.org/10.1038/s41598-021-95004-8) | SIVEP-Gripe, n = 231.112, com UF, IDH, escola, financiamento | XGBoost 0,813 (0,810–0,817) contra logística 0,766 (0,761–0,770). Fatores socioeconômicos e geográficos pesaram mais que comorbidade |
| [Lages dos Santos et al., *PeerJ Comput Sci* 2025](https://doi.org/10.7717/peerj-cs.2916) | SIVEP pediátrico. Eu não abri o texto completo; o resumo foi lido via a página do DOI | A logística foi a melhor, AUC 0,801, com gradient boosting e AdaBoost logo atrás. Saturação, comorbidade e idade |
| [Lima et al., *Rev Bras Saúde Mater Infant* 2021](https://doi.org/10.1590/1806-9304202100S200007) | Pernambuco, idosos, n = 7.486, sintomas e comorbidades, sem laboratório | Random forest, AUC 0,839. Idade, cardiopatia, saturação |
| [de Paiva et al., *Sci Rep* 2023](https://doi.org/10.1038/s41598-023-28579-z) | Registro brasileiro, n = 10.897, com sinal vital e exame | Stacking 0,826, LightGBM 0,824, LASSO 0,811. Rede neural 0,72–0,76. Os autores dizem que talvez já estejam no teto |
| [Ranzani et al., *Lancet Respir Med* 2021](https://doi.org/10.1016/S2213-2600(20)30560-9) | SIVEP, n = 254.288, descritivo | Mortalidade hospitalar 38%. Não é modelo preditivo. Serve de sanidade, e a minha prevalência não bate |
| Tese da FSP-USP, [página](https://teses.usp.br/teses/disponiveis/5/5137/tde-15102025-110217) | SIVEP de São Paulo. A página não abriu na leitura de 05/11; os números são os da busca | AUC 0,737–0,755 no geral, 0,824 nas primeiras semanas com 2.690 registros, caindo depois. Deriva temporal importa |

[Christodoulou et al., *J Clin Epidemiol* 2019](https://doi.org/10.1016/j.jclinepi.2019.02.004) revisaram comparações clínicas e não acharam ganho consistente de machine learning sobre logística. No Baqui o XGBoost ganha 0,047, mas o n é de centenas de milhares e há variável contínua e geográfica de verdade. No pediátrico do PeerJ, a logística ganhou. Com 6 mil linhas e quase tudo binário, eu esperava logística e boosting colados. Foi o que medi: 0,734 e 0,737.

Não achei competição ou notebook de referência no Kaggle para mortalidade com SIVEP. Não cito repositório solto que eu não li.

## Por que esta arquitetura

Tabelas pequenas continuam com árvore na frente de rede genérica ([Grinsztajn, Oyallon e Varoquaux, NeurIPS 2022](https://arxiv.org/abs/2207.08815)). Tuning leve de gradient boosting importa mais do que a briga entre rede e boosting, e o TabPFN aparece forte quando a amostra é da ordem de milhares ([McElfresh et al., NeurIPS 2023](https://arxiv.org/abs/2305.02997)). O TabPFN v2, no paper da [Nature 2025](https://doi.org/10.1038/s41586-024-08328-6), fica na frente em bases de até 10 mil linhas. O [TabArena](https://arxiv.org/abs/2506.16791) ([site](https://tabarena.ai)) diz que boosting segue competitivo, que modelo de fundação vai bem em amostra pequena, e que ensemble entre famílias ajuda, com o aviso de overfitting de conjunto de validação. O [relato do TabPFN-2.5](https://arxiv.org/abs/2511.08667) é da própria empresa; eu leio como conflito de interesse, não como auditoria. O CatBoost trata categórica sem o viés do encoding ingênuo ([Prokhorenkova et al., NeurIPS 2018](https://arxiv.org/abs/1706.09516)). O EBM da InterpretML é um GAM competitivo e legível ([Nori et al., 2019](https://arxiv.org/abs/1909.09223)).

Decisões, com o motivo:

1. **Logística com spline de idade, L2, C = 1, sem `class_weight`.** É o baseline que a literatura brasileira mostra competitivo, é o primeiro método que o regulamento cita, e o coeficiente das binárias ainda dá para ler. A spline existe porque a mortalidade sobe e achata. Uso a B-spline do scikit-learn (5 nós quantílicos, grau 3, sem coluna de viés porque a logística já tem intercepto, extrapolação linear fora dos nós). É o substituto prático de uma spline cúbica natural, sem depender do patsy. Não padronizei a matriz: binária e spline já estão em escala parecida, e padronizar a binária estraga a leitura do log-odds. A referência de cada nominal é a moda do treino do fold (no arquivo inteiro, 26 e 1). Categoria nova vira a referência.
2. **CatBoost com os defaults que importam explícitos:** `Logloss`, AUC na parada, profundidade 6, learning rate 0,05, `l2_leaf_reg` 3, até 1.000 árvores. A parada olha 20% do treino do fold; em seguida eu reajusto nesse treino inteiro com o número de árvores escolhido (média de 146 árvores nos 25 folds, variando de 37 a 352). `one_hot_max_size=10` deixa a nominal de 5 níveis em one-hot e a de 27 na estatística ordenada. `thread_count=4` e semente fixa. Neste ambiente o ajuste repetiu bit a bit; o arquivo gravado é o que vale se um retreino futuro divergir.
3. **Média 0,5 / 0,5, sem peso aprendido.** O TabArena alerta para overfitting de peso de ensemble. A faixa de empate é 0,010. Um peso estimado no mesmo OOF que eu vou citar não se paga. A regra de exceção, também anterior ao resultado: se a diferença de AUC passasse de 0,02, eu descartava o pior; se a média ficasse mais de 0,005 abaixo da idade, eu levava só a idade. Medido: logística 0,734, CatBoost 0,737, média 0,737, idade 0,717. DeLong pareado, média menos CatBoost: +0,0003, IC −0,001 a +0,002, p = 0,69. Média menos logística: +0,003, IC +0,001 a +0,005, p = 0,0002. Média menos idade: +0,020, IC +0,015 a +0,026. A média fica. Não porque ganhou uma corrida, e sim porque a regra disse para ficar e o número não a derrubou.
4. **Sem TabPFN no modelo que viaja.** O [README oficial](https://github.com/PriorLabs/TabPFN) diz que os pesos 2.5 em diante têm licença não comercial e exigem login na Prior Labs; o v2 é Apache com atribuição extra; em CPU, TabPFN-3 e 3.5 aceitam até 5.000 linhas e as versões anteriores até 1.000. Eu tenho 6.332, não tenho GPU neste ambiente, e no dia da prova não posso depender de download nem do cliente em nuvem. Um ensemble que quebra sem peso externo não serve. A logística e o CatBoost já estão a 0,003, dentro da faixa de 0,010 e dentro do desvio que o teste de 1.605 linhas tem. Não há, nos números que eu medi, um buraco que justifique o risco.
5. **Sem EBM.** Seria um terceiro ajuste, mais uma dependência, para um GAM que a logística com spline já aproxima. A banca vê a curva de idade, o coeficiente das binárias e a importância do CatBoost. Se a AUC da logística tivesse ficado muito atrás, eu reabriria isso. Não ficou.
6. **Sem rede, sem SVM, sem floresta extra.** O regulamento lista esses métodos. Rede, no de Paiva, ficou atrás. Floresta e SVM não tratam a nominal de 27 níveis melhor do que o CatBoost, e eu não ia multiplicar modelo dentro da mesma faixa de AUC.
7. **Sem SMOTE, sem undersampling, sem `class_weight`.** Isso desloca a probabilidade. A AUC não pede. O limiar é que carrega a preferência por sensibilidade. O slope ficou em 1,02 e o intercepto em −0,005, Brier 0,204, ECE por quantis 0,012. Calibração já está na diagonal. Não apliquei Platt. Não apliquei isotônica, que é degrau e pode derrubar a AUC ([Van Calster et al., *BMC Med* 2019](https://doi.org/10.1186/s12916-019-1466-7)).
8. **Hiperparâmetro não foi escolhido na CV que eu reporto.** [Varma e Simon, *BMC Bioinformatics* 2006](https://doi.org/10.1186/1471-2105-7-91) mostram o viés de escolher o modelo na mesma validação que estima o erro. Com a faixa de empate em 0,010, caçar 0,005 de AUC é ruído. A parada do CatBoost é interna ao fold, então o número de árvores não usa o pedaço que entra na AUC.

O `predict` do teste é a média dos modelos de fold, não um retreino em 100% das linhas. O limiar foi calibrado nos escores out-of-fold desses modelos. Retreinar em tudo mudaria a escala e o corte deixaria de valer. É a opção (a) da leitura de 05/11. A opção de exprimir o corte como quantil do teste usaria a distribuição do teste, e o regulamento pede um limiar em probabilidade. Não fiz.

## Avaliação

- `RepeatedStratifiedKFold`, 5×5, semente 42, os mesmos índices para idade, logística e CatBoost.
- AUC principal: uma probabilidade por paciente, média das cinco repetições out-of-fold, comparada ao rótulo. Não é a média das AUCs de fold, que esconde o fato de os folds serem pequenos.
- IC bootstrap estratificado, 2.000 réplicas, percentil 2,5 e 97,5. IC de DeLong no mesmo escore ([DeLong, DeLong e Clarke-Pearson, *Biometrics* 1988](https://doi.org/10.2307/2531595)). Os dois bateram na terceira casa. Comparação entre modelos é DeLong pareado da diferença.
- O desvio entre repetições (0,001 no ensemble) não é intervalo de confiança. Cada repetição reutiliza os mesmos pacientes. de Paiva lembra, citando Bengio e Grandvalet, que não há estimador não viesado da variância da validação cruzada. Eu trato o IC como aproximação da incerteza do escore out-of-fold, não como previsão da nota no teste.
- Calibração: Brier, intercepto e slope no logit, ECE em quantis, gráfico. Está no `metrics.json` e em `figs/calibracao.png`.
- Nada do teste entra no treino. O teste ainda nem existe. Quando existir, a imputação e as categorias vêm do fold já ajustado, e estatística do `teste.csv` não é calculada.

O roteiro do que reportar segue o [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378): população, desfecho, preditores, amostra, ausentes, modelo, validação, calibração, limiar, limitações. Eu não preenchi os 27 itens como submissão de revista. Usei a lista para não esquecer calibração e limiar no documento da banca.

## Limiar

A simulação binormal (AUC 0,80, prevalência 0,35, limiar escolhido em 6.332 escores e avaliado em 2.000 testes de 1.605) está em `scripts/simulacao_limiar.py`, semente 0. Rodei de novo. Os números batem com a tabela da leitura de 05/11, no arredondamento de duas ou três casas. Desvio-padrão da AUC no teste simulado: **0,0115**. IC empírico de 95%: 0,778 a 0,822. A faixa de empate de 0,010 é menor que um desvio-padrão. Diferença de 0,005 na minha CV não escolhe modelo.

| Regra na simulação | LR− no teste, mediana [p5–p95] | F1 mediano | Previstos como óbito |
| --- | --- | ---: | ---: |
| Youden | 0,44 [0,40–0,49] | 0,64 | 37% |
| Máximo F1 | 0,36 [0,31–0,40] | 0,65 | 46% |
| Sensibilidade ≥ 0,90 | 0,22 [0,17–0,26] | 0,62 | 66% |
| Sensibilidade ≥ 0,95 | 0,16 [0,11–0,21] | 0,59 | 77% |
| Sensibilidade ≥ 0,98 | 0,11 [0,07–0,17] | 0,57 | 85% |
| Sensibilidade ≥ 0,99 | 0,08 [0,04–0,15] | 0,55 | 91% |
| Sensibilidade 1 | 0,00 [0,00–0,12] | 0,52 | 99% |

Isso é ilustração, com prevalência 35% e AUC 0,80. A minha base tem prevalência 57% e AUC 0,74, então a LR− no ponto alto de sensibilidade fica pior do que a da tabela. Youden continua sendo um mau corte para este ranking.

Regra que entrou no código antes do resultado, e que eu não afrouxei depois:

- grade de quantis da probabilidade, de 1% a 99%;
- minimizar o p75 bootstrap da LR−;
- com pelo menos 8% previstos como 0;
- e F1 no máximo 0,08 abaixo do máximo F1 da grade;
- empate: maior F1, maior especificidade, limiar maior;
- se a grade não tiver candidato, há uma escada de folga (5% previstos como 0, depois sem piso de F1, depois maior especificidade entre os cortes com sensibilidade ≥ 0,95). Especificidade zero nunca é aceita.

O `TunedThresholdClassifierCV` do scikit-learn faz um serviço parecido ([documentação](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html)). Eu escrevi o meu porque a restrição é a LR− com dois pisos e o p75, e eu queria o corte gravado ao lado dos folds, não dentro de um estimador que reajusta na hora do `predict`.

O que saiu, no escore out-of-fold do ensemble:

| Corte | Limiar | Sens. | Espec. | LR− | F1 | Previstos 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Youden | 0,564 | 0,687 | 0,680 | 0,460 | 0,713 | 47,0% |
| Máximo F1 | 0,392 | 0,881 | 0,401 | 0,298 | 0,757 | 24,0% |
| Sens. ≥ 0,95 | 0,310 | 0,950 | 0,214 | 0,232 | 0,749 | 12,0% |
| **Regra (piso de 8%)** | **0,276163** | **0,969** | **0,145** | **0,214** | **0,743** | **8,0%** |
| Sens. ≥ 0,98 | 0,245 | 0,982 | 0,093 | 0,189 | 0,739 | 5,0% |
| Sens. ≥ 0,99 | 0,220 | 0,990 | 0,057 | 0,169 | 0,735 | 3,0% |

O piso de 8% é que segurou. O F1 ficou 0,014 abaixo do máximo, dentro da folga de 0,08. Ir para sensibilidade 0,98 melhoraria a LR− em 0,025 neste OOF e deixaria a especificidade em 0,09. No teste, com 1.605 linhas e poucos verdadeiros negativos, isso chega perto do infinito da regra. Não mexi.

O ponto 0,214 / 0,743 é otimista: o corte viu esses escores. Deixar uma repetição de fora deu LR− média **0,221** (mínimo 0,207, máximo 0,233) e F1 médio **0,744**, mas a repetição deixada de fora ainda contém os mesmos pacientes. O rótulo deles entra na escolha do corte. Não chamo isso de validação aninhada. O limiar único do arquivo continua 0,276163. A matriz nesse corte: VP 3.511, FN 113, FP 2.314, VN 394.

Para a banca, a frase é direta. O regulamento pede para não deixar óbito passar, então o corte é baixo. O freio de 8% existe para a especificidade não zerar num teste azarado. Não é um limiar de ambulatório.

## O que eu não confirmei

- A fonte oficial. SIVEP-Gripe é inferência.
- O mapa de `sociodemografico2`, de `sociodemografico5`, da 14ª clínica e de quais 7 comorbidades ficaram. Também não sei se óbito por outras causas (`EVOLUCAO` 3) foi excluído ou juntado ao óbito.
- Se o código 1–27 de `sociodemografico4` é UF, município agregado ou outra nominal com 27 níveis.
- A tese da USP, além do que a busca mostrou, porque a página não abriu.
- O texto completo do PeerJ 2025, além do resumo.
- O ranking ao vivo do TabArena. Li o paper, não o site no dia do treino.
- Se a banca aplica critério qualitativo além das três métricas. O regulamento diz que a vencedora sai do desempenho na etapa final, sem detalhar o resto. Um corte que marca 92% como óbito é fiel à LR− e ruim como ferramenta clínica. Deixei o trade-off escrito em vez de fingir que 0,5 seria neutro.

Não há assertiva de AUC no teste. Quando o `teste.csv` chegar, o comando é um só, e o limiar não será reestimado.

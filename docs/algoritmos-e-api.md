# Algoritmos e API

Camada de inteligência comportamental. Recebe o histórico de compras de um cliente e devolve um sinal: se merece atenção, com qual grupo se parece, se o valor sobe ou cai, e por quê.

A API não guarda clientes. Cada chamada é independente: o cálculo usa só os pedidos enviados naquela requisição e os modelos já treinados no servidor.

O identificador (`customer_id`) volta na resposta e não entra em nenhum cálculo.

## O que a API responde

Dado o histórico, a leitura junta quatro coisas:

| Campo | Pergunta | Valores |
| --- | --- | --- |
| `risk_level` | Precisa de atenção? | `low`, `medium`, `high` |
| `segment` | Com qual grupo se parece? | `new`, `potential`, `loyal`, `champion`, `at_risk` |
| `purchase_trend` | O valor das compras está indo para onde? | `growing`, `stable`, `declining` |
| `recommended_action` | O que fazer com isso? | ver tabela de ações abaixo |

`churn_risk` (0 a 1, duas casas) existe para **ordenar** clientes. Ele é a saída do modelo de floresta aleatória. Não é uma probabilidade calibrada de perda e não deve ser mostrado como porcentagem.

`customer_value` classifica o total gasto (`low`, `medium`, `high`) com cortes medidos na base em que o modelo foi treinado. `reasons` explica a leitura em frases. `confidence` diz o quanto o histórico sustenta essa leitura.

## Endpoints

Autenticação: header `X-API-Key`. A chave vem da variável de ambiente `API_KEY`. Sem chave configurada no servidor, a leitura responde 500. Chave ausente ou diferente responde 403. A comparação da chave é em tempo constante.

| Método | Rota | Chave | Função |
| --- | --- | --- | --- |
| `GET` | `/` | não | Confirma que o endereço abre. Devolve `status` e `version`. |
| `GET` | `/health` | não | Confirma que o processo está respondendo (`healthy`). |
| `POST` | `/analyze` | sim | Lê um cliente. |
| `POST` | `/batch` | sim | Lê vários clientes na mesma chamada. |

### `POST /analyze`

Corpo:

```json
{
  "customer_id": "ana",
  "orders": [
    {"date": "2026-03-10", "value": 300},
    {"date": "2026-04-15", "value": 350}
  ]
}
```

Resposta (`CustomerIntelligence`):

| Campo | Significado |
| --- | --- |
| `customer_id` | O mesmo identificador enviado. |
| `churn_risk` | Score de 0 a 1 para ordenar. Quanto maior, mais o modelo associa o padrão a quem deixou de comprar. |
| `risk_level` | Faixa de atenção derivada do score. |
| `segment` | Grupo do K-Means. |
| `purchase_trend` | Direção do valor ao longo das compras. |
| `customer_value` | Faixa do total gasto nesta base. |
| `recommended_action` | Ação recomendada. |
| `reasons` | Frases com a métrica que sustentou a leitura. |
| `confidence` | `low` (1–2 compras), `medium` (3–4), `high` (5 ou mais). |
| `value_high_from` | Total a partir do qual o gasto é alto nesta base. |
| `value_medium_from` | Total a partir do qual o gasto é médio nesta base. |

Lista vazia de pedidos responde 400. Falha inesperada no cálculo responde 500, sem detalhe interno.

### `POST /batch`

O corpo é uma lista de clientes, no mesmo formato de `/analyze`.

- Até 500 clientes.
- Até 100.000 pedidos somados na chamada.
- Lista vazia, ou um desses tetos estourado, responde 400.
- Um cliente com problema volta com `error`. Os outros seguem. Um erro não cancela o lote.
- Cliente sem pedidos volta com `error` e sem `analysis`.

### O que a entrada recusa

Validação do pedido (resposta 422):

- `date` precisa ser `YYYY-MM-DD` (ISO). Data mais de 1 dia à frente de hoje é recusada. O dia de folga cobre diferença de fuso entre a loja e o servidor.
- `value` menor que zero é recusado. Devolução e estorno não entram como compra.
- Mais de 5.000 pedidos por cliente é recusado.

Valor zero passa na API. No treino por CSV, valor zero, ausente ou negativo é rejeitado.

## Como uma leitura é montada

```
pedidos
  → features (recência, frequência, valor, tendência, intervalo)
  → floresta aleatória → churn_risk → risk_level
  → K-Means → segment
  → regras → purchase_trend, customer_value, recommended_action, reasons
```

Treino e leitura usam a mesma lista de features, na mesma ordem:

`recency`, `frequency`, `monetary_avg`, `monetary_total`, `trend_slope`, `avg_days_between`.

O K-Means usa só as quatro primeiras. Tendência e intervalo ficam de fora do agrupamento.

Na API, “hoje” é o instante da chamada. No treino, “hoje” é a data de corte: as features enxergam só compras anteriores a esse dia.

## 1. Features do histórico

Arquivo: `app/ml/features.py`. Não é um modelo. É o retrato numérico de um cliente, calculado só com os pedidos recebidos.

Os pedidos são ordenados por data antes de qualquer conta.

### Recência (`recency`)

Dias inteiros entre a compra mais recente e a data de referência (agora, na API; a data de corte, no treino). Número alto significa que a pessoa sumiu faz tempo.

### Frequência (`frequency`)

Quantidade de pedidos na lista. Na API, cada item de `orders` conta como um evento. No CSV de treino, compras do mesmo cliente no mesmo dia são somadas num único pedido antes do cálculo.

### Valor (`monetary_avg` e `monetary_total`)

Média e soma de `value`. O total alimenta `customer_value`. A média alimenta a razão de ticket baixo.

### Tendência (`trend_slope`)

Com 2 ou mais pedidos, é a inclinação de uma reta ajustada aos valores na ordem das compras (`numpy.polyfit` de grau 1). O eixo x é o índice da compra (0, 1, 2…), não o calendário. A inclinação é quanto o valor muda, em média, de uma compra para a seguinte.

Inclinação positiva: os valores sobem ao longo do histórico. Negativa: caem. Com um único pedido a inclinação é 0, porque não há direção para medir.

### Intervalo (`avg_days_between`)

Média, em dias, dos espaços entre compras consecutivas. Com um único pedido o valor é 0 e significa “desconhecido”, não “comprou todo dia”.

### Baseline individual

Com 3 ou mais pedidos, o intervalo habitual (`expected_interval`) é essa média, e o desvio é `recency - expected_interval`. A explicação compara a pessoa com o próprio ritmo, em vez de um prazo fixo igual para todo mundo.

Com menos de 3 pedidos não há baseline. A razão de sumiço, se houver, usa um fallback: mais de 60 dias desde a última compra.

### Confiança

Depende só da quantidade de eventos:

| Compras | `confidence` |
| --- | --- |
| 1 ou 2 | `low` |
| 3 ou 4 | `medium` |
| 5 ou mais | `high` |

Confiança baixa significa histórico curto. O padrão ainda não está firme.

## 2. Floresta aleatória (risco)

Arquivo de treino: `app/ml/train.py`. Leitura: `app/models/predictor.py`.

Algoritmo: `RandomForestClassifier` do scikit-learn, dentro de um `Pipeline` com `StandardScaler`.

| Parâmetro | Valor |
| --- | --- |
| Árvores | 200 |
| Mínimo de amostras por folha | 5 |
| `random_state` | 42 |

O scaler fica dentro do pipeline. Treino e leitura normalizam do mesmo jeito. A floresta recebe as seis features. A classe positiva é “churn”.

### O que o rótulo significa no treino

Para cada data de corte, o rótulo olha só o que aconteceu **depois**:

- features = compras **antes** do corte;
- `churn = 1` se a pessoa **não** comprou nos 90 dias seguintes;
- `churn = 0` se comprou nesse horizonte.

O último corte temporal mede o modelo e não entra no ajuste. Isso evita que o teste veja o mesmo período que o treino.

### O que sai na API

`predict_proba` da classe 1, arredondado para duas casas, vira `churn_risk`.

Faixas fixas, independentes da base:

| Score | `risk_level` | Leitura |
| --- | --- | --- |
| abaixo de 0,35 | `low` | pode esperar |
| de 0,35 até abaixo de 0,65 | `medium` | olhar |
| 0,65 ou mais | `high` | vale chamar |

O número serve para colocar um cliente na frente do outro. A faixa é o que se mostra para uma pessoa. O modelo não foi calibrado: 0,70 não significa “70% de chance de perder o cliente”.

## 3. K-Means (segmento)

Arquivo: `app/ml/segmenter.py`.

Agrupa clientes parecidos em 5 grupos. Features: recência, frequência, ticket médio e total. `StandardScaler` e depois `KMeans` (`n_clusters=5`, `n_init=10`, `random_state=42`).

O ajuste usa o retrato do **último corte de treino**, um ponto por cliente. O mapa `cluster_id → nome` é gravado junto com o pipeline.

Os nomes não vêm do algoritmo. Depois do ajuste, os centróides voltam à escala original e recebem rótulo nesta ordem:

1. Menor frequência média → `new`.
2. Entre os que sobraram, maior recência média → `at_risk`.
3. Os três restantes, do maior ticket médio para o menor → `champion`, `loyal`, `potential`.

| Segmento | Ideia |
| --- | --- |
| `new` | Poucas compras. Histórico ainda curto. |
| `at_risk` | Mais tempo sem comprar do que os outros grupos. |
| `potential` | Já compra, com ticket abaixo de leal e campeão. |
| `loyal` | Ticket intermediário entre potencial e campeão. |
| `champion` | Maior ticket médio entre os três grupos que restaram. |

Se um cluster não estiver no mapa gravado, a leitura devolve `potential`.

O segmento descreve com quem a pessoa se parece na base de treino. Ele não substitui o risco. Os dois podem discordar; a ação recomendada resolve isso (abaixo).

## 4. Regras em cima das features

Arquivo: `app/models/predictor.py`. Não são modelos. Traduzem números em tendência, valor, ação e frases.

### Tendência de compra

Compara `trend_slope` com `trend_delta` da base treinada.

| Condição | `purchase_trend` |
| --- | --- |
| inclinação maior que `+trend_delta` | `growing` |
| inclinação menor que `−trend_delta` | `declining` |
| no meio | `stable` |

`trend_delta` é 10% do ticket mediano da base, com piso de 1. Assim uma oscilação pequena não vira “crescendo” ou “caindo”.

### Valor do cliente

Compara `monetary_total` com percentis da base de treino (último corte):

| Condição | `customer_value` |
| --- | --- |
| total ≥ percentil 75 | `high` |
| total ≥ percentil 40 | `medium` |
| abaixo disso | `low` |

Se o percentil 40 ficar igual ou acima do 75, o corte médio vira metade do corte alto. Os dois cortes voltam na resposta em `value_high_from` e `value_medium_from`, na mesma unidade dos pedidos (reais, libras, o que a base usar).

Sem arquivo de limiares, os fallbacks são: alto a partir de 3000, médio a partir de 800, ticket baixo abaixo de 100, tendência ±10.

### Ação recomendada

O risco tem prioridade quando discorda do segmento. Isso evita tratar como cliente novo alguém que já comprou e está sumindo.

| Condição | `recommended_action` |
| --- | --- |
| `risk_level` alto e 2 ou mais compras | `retention` |
| `risk_level` baixo e segmento `at_risk` | `monitor` |
| segmento `champion` | `maintain_engagement` |
| segmento `loyal` | `maintain_relationship` |
| segmento `at_risk` | `retention` |
| segmento `new` | `onboarding` |
| segmento `potential` | `nurture` |
| segmento desconhecido | `monitor` |

### Razões

Cada frase cita a métrica. A lista pode ter mais de uma. Se nenhuma regra dispara, a razão é “comportamento dentro do padrão esperado”.

| Regra | Frase |
| --- | --- |
| Com baseline (3+ compras) e recência maior que 1,5× o intervalo habitual | Dias desde a última atividade, o intervalo habitual e quantas vezes acima do esperado. |
| Sem baseline e recência acima de 60 dias | Dias desde a última atividade, com aviso de histórico curto. |
| Menos de 3 compras | Padrão ainda não estabelecido. |
| Tendência `declining` | Valor das transações em queda. |
| Ticket médio abaixo do percentil 25 da base (`ticket_low`) | Ticket médio baixo, com o valor. |

## 5. Como o modelo é treinado

Comando, a partir de um CSV com `customer_id`, `date` e `value` (também aceita aliases como `cliente`, `data`, `valor`):

```bash
python -m app.ml.train --orders pedidos.csv
```

O CSV pode usar vírgula ou ponto e vírgula. Datas inválidas, valores não positivos e linhas sem cliente interrompem o treino. Compras do mesmo cliente no mesmo dia viram um pedido (soma dos valores).

### Cortes temporais

O horizonte é 90 dias. O treino escolhe inícios de mês em que ainda cabem 90 dias de futuro dentro da base, e fica com um corte a cada trimestre. O último corte é só teste.

Em cada corte de treino:

1. monta as features com compras anteriores;
2. marca churn conforme os 90 dias seguintes;
3. ajusta a floresta em todos os cortes de treino juntos.

O K-Means e os limiares usam apenas o retrato do último corte de treino (um cliente, um ponto).

### Quando o artefato é gravado

Contar clientes não aprova o modelo. Os dois modelos e os limiares só substituem os arquivos se **todas** as condições passarem no corte de teste:

| Medida | Mínimo |
| --- | --- |
| Clientes no treino | 200 |
| Clientes no teste | 200 |
| Taxa de churn no teste | entre 5% e 95% |
| ROC AUC | 0,65 |
| Acurácia acima do chute da classe mais comum | 5 pontos percentuais |
| Clientes em cada segmento (`new`, `potential`, `loyal`, `champion`, `at_risk`) | 50 |

Se alguma falhar, o artefato anterior permanece. A API precisa ser reiniciada para carregar um artefato novo: modelo, segmentador e limiares ficam em memória depois da primeira leitura.

Arquivos gravados em `app/ml/`:

- `model.joblib` — floresta aleatória
- `segmenter.joblib` — pipeline do K-Means
- `segment_map.joblib` — id do cluster para o nome do segmento
- `thresholds.joblib` — cortes de valor, ticket e tendência

### Benchmark e dados sintéticos

```bash
python -m app.ml.train --benchmark   # Online Retail II, só comparação
python -m app.ml.train --synthetic   # fórmula artificial, sem corte temporal
```

`--benchmark` baixa o Online Retail II (UCI, loja do Reino Unido, dez/2009 a dez/2011, valores em GBP, muitos clientes atacadistas). Passa pelo mesmo critério de gravação. Métricas desse conjunto medem aquele dataset.

`--synthetic` gera features com uma fórmula (recência alta e frequência baixa aumentam o rótulo de churn) e um split aleatório 80/20. Não há corte temporal. A acurácia só mostra que a floresta copiou a fórmula. Não é evidência sobre clientes reais.

O `model.joblib` que acompanha o repositório é o treino do Online Retail II. Ele deixa de ser o modelo quando uma base própria passa na medição.

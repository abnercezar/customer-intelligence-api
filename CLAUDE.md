# Customer Behavioral Intelligence API

## O que é este projeto

Uma camada de inteligência comportamental que analisa o histórico de eventos de clientes e devolve sinais acionáveis para qualquer tipo de negócio. Não é um CRM.

**Pergunta central que o produto responde:**
> "O que está acontecendo com meus clientes e quais precisam da minha atenção?"

## Localização

```
C:\Users\abner\customer-intelligence-api\
```

## Stack

- Python 3.11
- FastAPI + Pydantic
- scikit-learn (RandomForest + KMeans)
- pandas / numpy
- Streamlit (dashboard)
- pytest

## Estrutura de arquivos

```
app/
  main.py              — endpoints FastAPI (/analyze, /batch, /health)
  security.py          — autenticação via API Key (X-API-Key header)
  ml/
    features.py        — extração de features RFM (recency, frequency, monetary, trend, interval)
    train.py           — treino: RandomForest (churn) + KMeans (segmentação)
    segmenter.py       — K-Means: agrupa e rotula clusters automaticamente
    model.joblib       — modelo de churn treinado
    segmenter.joblib   — pipeline K-Means treinado
    segment_map.joblib — mapeamento cluster_id → nome do segmento
  models/
    schemas.py         — Pydantic: CustomerRequest, CustomerIntelligence, BatchItemResult
    predictor.py       — lógica completa: features → churn → segmento → razões → resposta
dashboard.py           — interface Streamlit local
tests/
  test_api.py          — 12 testes cobrindo auth, analyze, batch, edge cases
data/
  sample_customers.json
requirements.txt
```

## Fluxo de dados

```
POST /analyze
  └── CustomerRequest (customer_id + orders[])
        └── predictor.predict()
              ├── features.extract_features()     → recency, frequency, monetary, trend, interval
              ├── model.predict_proba()            → behavior_score (0–1)
              ├── segmenter.predict_segment()      → champion | loyal | potential | at_risk | new
              ├── _trend()                         → growing | stable | declining
              ├── _value()                         → high | medium | low
              ├── _action()                        → ação recomendada
              └── _reasons()                       → lista de razões explicáveis
```

## Sinais do produto

| Sinal | Ação |
|-------|------|
| new | onboarding |
| healthy | maintain_engagement |
| loyal | maintain_relationship |
| potential | nurture |
| at_risk | retention |
| reactivation | reactivation |
| growth_opportunity | expansion |

A ação vem do segmento, mas o risco do modelo tem prioridade quando discordam: `risk_level` alto com 2+ compras → `retention`; `at_risk` com risco baixo → `monitor`.

## Contrato da API — mudanças

- `risk_level` (`low` < 0.35 ≤ `medium` < 0.65 ≤ `high`) foi adicionado à resposta. É o que deve ser mostrado a pessoas; `churn_risk` continua para ordenar clientes, mas não é probabilidade calibrada e não deve aparecer como %.
- Entradas recusadas com 422: valor negativo (devolução/estorno), data mais de 1 dia no futuro, mais de 5000 pedidos por cliente.
- `/batch` recusa com 400 mais de 100.000 pedidos somados por chamada.

## Estado atual do modelo

O motor treina numa base de pedidos (`python -m app.ml.train --orders arquivo.csv`): mesmas features para o RandomForest e para o K-Means, limiares de valor e tendência medidos nessa base, último corte só para teste. O artefato só é gravado se a medição passar (AUC, ganho sobre o baseline, tamanho dos segmentos). O score não é probabilidade calibrada.

O arquivo `model.joblib` que está no repositório ainda é o treino do Online Retail II (benchmark, Reino Unido, 2009–2011, GBP). Ele deixa de ser o modelo quando uma base passar na medição. O Online Retail II continua disponível com `--benchmark`.

- Rótulo de churn: features só com pedidos antes do corte; churn = não comprou nos 90 dias seguintes
- No benchmark, a avaliação temporal antiga (teste em 2011-09) deu acurácia 72,2% vs. baseline 57,3%, ROC AUC 0,786 — isso mede aquele dataset, não clientes do produto

## Regras de desenvolvimento

1. Ler o código antes de modificar
2. Não recriar o que já existe
3. Não refatorar sem necessidade clara
4. Não alterar contratos de API sem documentar impacto
5. Não apagar funcionalidade sem confirmação
6. Todo código novo tem teste
7. Não inventar dados nem apresentar sintéticos como reais
8. Não chamar heurística de IA nem score de probabilidade calibrada sem validação
9. Priorizar explicabilidade
10. Preferir a solução mais simples

## Critério de sucesso

> Dado o histórico de um cliente, o sistema identifica uma mudança relevante no comportamento e explica claramente por que ela merece atenção.

## Rodar localmente

```bash
# Treinar churn, segmentos e limiares num CSV da base (customer_id, date, value)
python -m app.ml.train --orders pedidos.csv

# Benchmark: Online Retail II (não substitui o modelo se a medição falhar)
python -m app.ml.train --benchmark

# Dados sintéticos, sem corte temporal
python -m app.ml.train --synthetic

# API
uvicorn app.main:app --reload

# Dashboard
streamlit run dashboard.py --browser.gatherUsageStats false

# Testes
python -m pytest tests/ -v
```

## Variáveis de ambiente

```
API_KEY=sua-chave-aqui
API_URL=http://127.0.0.1:8000/analyze  (dashboard)
```

# Como rodar o projeto

Passo a passo no Windows (PowerShell), do terminal fechado até a API e o dashboard funcionando.

> Use sempre o Python do `.venv` (`.\.venv\Scripts\python.exe`), nunca o `python` global.
> O Python global tem pacotes de outros projetos com versões incompatíveis.

---

## 1. Abrir o terminal na pasta do projeto

Abra o PowerShell (tecla Windows → digite `PowerShell` → Enter) e entre na pasta:

```powershell
cd C:\Users\abner\customer-intelligence-api
```

---

## 2. Preparar o ambiente (só na primeira vez)

Pule esta etapa se a pasta `.venv` já existir.

```powershell
# Cria o ambiente virtual do projeto
python -m venv .venv

# Instala as dependências
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sempre que o `requirements.txt` mudar, rode o `pip install` de novo.

---

## 3. Treinar os modelos (opcional)

Os modelos treinados já estão no projeto (`app/ml/*.joblib`). Só retreine se mudar algo no ML.

```powershell
# Modelo de churn com dados reais (Online Retail II)
# Na 1ª vez baixa o dataset (~45 MB) para data/raw/ e demora alguns minutos
.\.venv\Scripts\python.exe -m app.ml.train

# Modelo de churn + segmentador com dados sintéticos
.\.venv\Scripts\python.exe -m app.ml.train --synthetic
```

> Atenção: `--synthetic` também sobrescreve o modelo de churn com a versão sintética.
> Para voltar ao modelo real, rode depois `.\.venv\Scripts\python.exe -m app.ml.train`.

---

## 4. Rodar os testes

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Todos devem passar.

---

## 5. Subir a API

No terminal da etapa 1:

```powershell
$env:API_KEY = "teste123"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Espere aparecer `Uvicorn running on http://127.0.0.1:8000`.
**Deixe este terminal aberto** — fechar ou apertar Ctrl+C derruba a API.

Teste no navegador:

- http://127.0.0.1:8000/health → deve mostrar `{"status":"healthy"}`
- http://127.0.0.1:8000/docs → documentação interativa
  1. Clique em **Authorize** (cadeado) e digite `teste123`
  2. Abra `POST /analyze` → **Try it out** → cole um cliente de `data/sample_customers.json` → **Execute**
  3. Para vários clientes, use `POST /batch` com o conteúdo inteiro de `data/sample_customers.json`

---

## 6. Subir o dashboard (opcional)

Abra um **segundo** PowerShell (a API precisa continuar rodando no primeiro):

```powershell
cd C:\Users\abner\customer-intelligence-api
$env:API_KEY = "teste123"
.\.venv\Scripts\python.exe -m streamlit run dashboard.py --browser.gatherUsageStats false
```

Acesse http://localhost:8501. A `API_KEY` precisa ser a mesma da etapa 5.

---

## 7. Parar tudo

Em cada terminal, aperte **Ctrl+C** (ou feche a janela).

---

## Problemas comuns

| Erro | Causa | Solução |
|------|-------|---------|
| `ERR_CONNECTION_REFUSED` em `:8000` | A API não está rodando | Refaça a etapa 5 e deixe o terminal aberto |
| `ERR_CONNECTION_REFUSED` em `:8501` | O dashboard não está rodando | Refaça a etapa 6 |
| `403` na resposta | Chave ausente ou diferente da configurada | Use a mesma `API_KEY` da etapa 5 no Authorize / dashboard |
| `500` com "API_KEY não configurada" | A API subiu sem `$env:API_KEY` | Pare a API (Ctrl+C) e refaça a etapa 5 |
| Dashboard mostra "API offline" | A API não está rodando | Suba a API (etapa 5) antes do dashboard |
| `No module named ...` | Dependências não instaladas | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| Porta já em uso (`address already in use`) | Outra instância ainda rodando | Feche o terminal antigo ou use `--port 8002` |
| Erros estranhos de `starlette` / `fastapi` | Rodou com o `python` global | Use `.\.venv\Scripts\python.exe` em vez de `python` |

---

## Variáveis de ambiente

| Variável | Onde | Exemplo |
|----------|------|---------|
| `API_KEY` | API e dashboard | `teste123` (em produção, gere uma forte: `.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"`) |
| `API_URL` | dashboard | `http://127.0.0.1:8000/analyze` (padrão) |

`$env:...` só vale para o terminal atual — em cada terminal novo, defina de novo.

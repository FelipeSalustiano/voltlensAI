# Monitoramento — VoltLens AI

Este documento explica como acompanhar a API de inferência por meio dos logs JSONL e das métricas de runtime disponibilizadas pelo BentoML. Ele diferencia o que **já está implementado** do que é apenas **recomendação para produção**.

## Sumário

1. [Visão geral](#1-visão-geral)
2. [Logs estruturados](#2-logs-estruturados)
3. [Métricas Prometheus](#3-métricas-prometheus)
4. [Logs versus métricas](#4-logs-versus-métricas)
5. [Retenção dos registros](#5-retenção-dos-registros)
6. [Fluxo recomendado de análise](#6-fluxo-recomendado-de-análise)

## 1. Visão geral

O projeto utiliza duas fontes complementares e um arquivo de snapshot das métricas:

| Fonte | Finalidade | Persistência |
| --- | --- | --- |
| `logs/inference.log` | Eventos individuais, metadados da imagem, saída, latência, status HTTP e erros | Arquivo local em modo append |
| `http://localhost:3000/metrics` | Contadores, requisições em andamento e distribuição agregada da duração | Memória do processo; o histórico depende de um coletor, como o Prometheus |
| `logs/metrics_predict.txt` | Cópia das métricas de `/predict` em um instante (veja a [seção 3.5](#35-snapshot-das-métricas-logsmetrics_predicttxt)) | Arquivo de texto; não é atualizado automaticamente |

Os logs permitem investigar requisições específicas. As métricas permitem acompanhar o comportamento agregado do serviço.

## 2. Logs estruturados

### 2.1 Arquivo e formato

Os eventos são gravados em `logs/inference.log` no formato **JSON Lines (JSONL)**: cada linha contém um objeto JSON independente. O arquivo é aberto em modo append, e novos registros são adicionados ao final.

Campos que um evento pode conter:

| Campo | Descrição |
| --- | --- |
| `timestamp` | Data e hora em UTC |
| `request_id` | Identificador da requisição |
| `endpoint` | Endpoint acessado (`/predict`) |
| `input` | Metadados da imagem |
| `output` | Resultado da inferência, quando disponível |
| `latency_ms` | Latência em milissegundos |
| `status_code` | Status HTTP |
| `error` | Descrição do erro, quando existente |

- **Eventos bem-sucedidos:** `input` inclui largura, altura, formato, média e desvio-padrão dos pixels; `output` inclui a indicação de detecção, a confiança e a bounding box.
- **Falhas:** `output` pode ser nulo, e o registro pode conter `content_type`, `content_length`, status e erro.

Para consultar as últimas linhas no PowerShell:

```powershell
Get-Content .\logs\inference.log -Tail 10
```

### 2.2 Análise com o notebook

O notebook `notebooks/voltlensai_log_analysis.ipynb` usa pandas para analisar volume de requisições, status HTTP, taxa de erro, latência, confiança, características das imagens, evolução temporal e eventos com erro.

Inicie o Jupyter:

```bash
uv run jupyter notebook
```

Em seguida, abra `notebooks/voltlensai_log_analysis.ipynb`.

Exemplo de leitura e normalização do JSONL:

```python
import json
import pandas as pd

with open("logs/inference.log", "r", encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]

df = pd.json_normalize(records, sep="_")
```

A normalização transforma campos aninhados em colunas, como `input_pixel_mean` e `output_confidence`.

### 2.3 Exportação CSV

O notebook permite exportar os registros normalizados para `logs/inference_analysis.csv`:

```python
from pathlib import Path

EXPORT_PATH = Path("../logs/inference_analysis.csv")
df.to_csv(EXPORT_PATH, index=False, encoding="utf-8-sig")
```

> **Atenção:** a exportação é manual. O CSV representa os registros existentes no momento em que a célula foi executada e **não é atualizado automaticamente** quando novas linhas são adicionadas ao JSONL. Para obter uma versão atualizada, execute novamente as células de leitura e exportação. O `inference.log` continua sendo a fonte original dos eventos.

## 3. Métricas Prometheus

### 3.1 Consultar o endpoint

Com o serviço em execução, acesse:

```text
http://localhost:3000/metrics
```

No PowerShell:

```powershell
curl.exe http://localhost:3000/metrics
```

Entre as métricas disponibilizadas pelo BentoML estão:

| Métrica | Tipo | Significado |
| --- | --- | --- |
| `bentoml_service_request_total` | Counter | Total acumulado de requisições |
| `bentoml_service_request_duration_seconds` | Histogram | Distribuição da duração das requisições |
| `bentoml_service_last_request_timestamp_seconds` | Gauge | Timestamp da última requisição |
| `bentoml_service_request_in_progress` | Gauge | Requisições em processamento |

O histograma possui séries auxiliares:

- `_bucket`: contagens acumuladas por limite superior (`le`);
- `_sum`: soma das durações;
- `_count`: quantidade de observações.

> **Confira os nomes e labels realmente exibidos no endpoint local**, pois podem variar conforme a versão do BentoML. Os exemplos abaixo pressupõem que `bentoml_service_request_total` exponha o label `http_response_code`. Se o nome do label for diferente, ajuste o seletor PromQL.

### 3.2 Taxa de erro

A taxa de erro é a proporção de requisições com status HTTP 4xx ou 5xx em relação ao total, dentro de um intervalo:

```text
Taxa de erro (%) = (requisições com erro / total de requisições) × 100
```

Consulta PromQL para os últimos cinco minutos:

```promql
100 *
sum(rate(bentoml_service_request_total{http_response_code=~"4..|5.."}[5m]))
/
sum(rate(bentoml_service_request_total[5m]))
```

- `rate(...[5m])` calcula a taxa média por segundo na janela.
- O seletor `4..|5..` filtra as respostas de erro.
- Sem tráfego na janela, o denominador é zero e o resultado aparece como `NaN`; isso é esperado.
- Se o contador não expuser um label de status HTTP, não é possível separar os erros com essa consulta. Nesse caso, use os logs ou implemente uma métrica específica por status.

Também é possível calcular a taxa a partir dos logs normalizados:

```python
total = len(df)
errors = df["status_code"].ge(400).sum()
error_rate = (errors / total) * 100 if total else 0
print(f"Taxa de erro: {error_rate:.2f}%")
```

### 3.3 Latência e percentis

A duração do histograma é expressa em **segundos**. A média dos últimos cinco minutos pode ser estimada por:

```promql
sum(rate(bentoml_service_request_duration_seconds_sum[5m]))
/
sum(rate(bentoml_service_request_duration_seconds_count[5m]))
```

O P95 pode ser estimado com:

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(bentoml_service_request_duration_seconds_bucket[5m])
  )
)
```

O P95 é o valor abaixo do qual está aproximadamente 95% das observações da janela. Por exemplo, `0.12` corresponde a cerca de 120 ms. Como o valor é interpolado a partir dos buckets, a precisão depende dos limites (`le`) configurados.

Para exibir o resultado em milissegundos, multiplique por 1000:

```promql
1000 *
histogram_quantile(
  0.95,
  sum by (le) (
    rate(bentoml_service_request_duration_seconds_bucket[5m])
  )
)
```

Para P50 ou P99, substitua `0.95` por `0.50` ou `0.99`.

### 3.4 Histograma e boxplot

O **histograma** mostra a frequência de observações em faixas de duração. Os buckets do Prometheus permitem acompanhar essa distribuição de forma agregada e estimar quantis.

O **boxplot** resume observações individuais por mediana, Q1, Q3, intervalo interquartil e valores extremos. Para construí-lo, use `latency_ms` dos logs ou do CSV, e não apenas as séries agregadas do Prometheus.

Exemplo com pandas e matplotlib, considerando apenas respostas HTTP 200:

```python
import matplotlib.pyplot as plt

successful = df.loc[
    df["status_code"].eq(200) & df["latency_ms"].notna(),
    "latency_ms"
]

plt.figure(figsize=(8, 4))
plt.boxplot(successful, vert=False)
plt.xlabel("Latência (ms)")
plt.title("Latência — requisições bem-sucedidas")
plt.grid(axis="x", alpha=0.3)
plt.show()
```

Para o histograma:

```python
plt.figure(figsize=(8, 4))
plt.hist(successful, bins=20)
plt.xlabel("Latência (ms)")
plt.ylabel("Frequência")
plt.title("Histograma de latência — requisições bem-sucedidas")
plt.grid(axis="y", alpha=0.3)
plt.show()
```

> **Atenção:** no caminho de sucesso, a latência é medida no fluxo de inferência. Para falhas capturadas pelo middleware, a medição considera o tempo observado no processamento da requisição ASGI. Os escopos são diferentes; por isso, analise as falhas separadamente e não misture as duas populações no mesmo boxplot.

### 3.5 Snapshot das métricas: `logs/metrics_predict.txt`

O arquivo `logs/metrics_predict.txt` é uma **cópia em texto das métricas do endpoint `/predict`**, obtida a partir de `/metrics` em um instante específico. Ele usa o formato de exposição do Prometheus: cada linha é uma amostra no padrão `nome{labels} valor`.

Ele serve para consultar ou comparar o estado das métricas sem precisar do serviço em execução. Atenção a duas características:

- **É um retrato de um momento.** O arquivo não é atualizado automaticamente; para obter valores novos, gere-o de novo (veja abaixo).
- **Os valores são acumulados desde o início do processo.** Se o serviço reiniciar, os contadores voltam a zero.

Ele contém apenas as séries de `bentoml_service_request_total`, `bentoml_service_request_in_progress` e `bentoml_service_request_duration_seconds`, sem as linhas `# HELP` e `# TYPE`. Isso indica que foi filtrado a partir da saída completa de `/metrics`. A métrica `bentoml_service_last_request_timestamp_seconds` não aparece nele.

#### Anatomia de uma linha

```text
bentoml_service_request_total{endpoint="/predict",http_response_code="200",runner_name="MeterDetectionService",service_name="MeterDetectionService",service_version="not available"} 90.0
```

| Parte | Significado |
| --- | --- |
| `bentoml_service_request_total` | Nome da métrica (contador de requisições) |
| `endpoint` | Endpoint medido (`/predict`) |
| `http_response_code` | Status HTTP da resposta; separa as séries de sucesso (`200`) e de erro (`500`) |
| `runner_name` e `service_name` | Nome do serviço BentoML (`MeterDetectionService`) |
| `service_version` | Versão do serviço; `not available` indica que não foi definida |
| `le` | Presente só nos buckets; limite superior da faixa de duração, em segundos |
| `90.0` | Valor da amostra |

O label `http_response_code` existe nesta versão, então as consultas de taxa de erro da seção 3.2 funcionam sem ajustes de seletor.

#### Leitura do exemplo

| Métrica | HTTP 200 | HTTP 500 |
| --- | --- | --- |
| `bentoml_service_request_total` | 90 | 10 |
| `bentoml_service_request_duration_seconds_count` | 90 | 10 |
| `bentoml_service_request_duration_seconds_sum` | 9,54 s | 0,53 s |

`bentoml_service_request_in_progress` vale `0.0`: nenhuma requisição estava em processamento no momento da captura.

A partir desses números:

- **Total de requisições:** 90 + 10 = 100.
- **Taxa de erro acumulada:** 10 / 100 = **10%**.
- **Latência média das respostas 200:** 9,54 / 90 ≈ 0,106 s (**≈ 106 ms**).
- **Latência média das respostas 500:** 0,53 / 10 ≈ 0,053 s (**≈ 53 ms**).

#### Como ler os buckets

Os buckets são **cumulativos**: cada linha `le="X"` conta as requisições com duração menor ou igual a X segundos. Nas respostas 200 do exemplo:

| Limite (`le`) | Requisições | Leitura |
| --- | --- | --- |
| `0.05` | 0 | Nenhuma terminou em até 50 ms |
| `0.1` | 27 | 27 terminaram em até 100 ms |
| `0.25` | 89 | 89 terminaram em até 250 ms |
| `0.5` | 90 | Todas terminaram em até 500 ms |

Como o P95 corresponde à requisição de posto ~85,5 (de 90), ele cai na faixa entre 0,1 s e 0,25 s. Interpolando linearmente dentro dessa faixa, a estimativa é de **≈ 0,24 s (≈ 241 ms)**. O P50 fica em **≈ 0,14 s (≈ 144 ms)**. São estimativas, e a precisão depende da largura dos buckets.

Nas respostas 500, 9 das 10 requisições terminaram em até 10 ms e uma levou entre 250 ms e 500 ms. Isso sugere que a maioria das falhas ocorre cedo no fluxo, antes da inferência, mas a causa só pode ser confirmada nos eventos de `logs/inference.log` (campos `error` e `request_id`).

#### Limitações

- **Sem janela de tempo.** O arquivo mostra totais acumulados, não taxas. Calcular `rate()` ou comparar períodos exige coletar `/metrics` periodicamente com o Prometheus.
- **O histograma é separado por status.** Consultas com `sum by (le)` somam as respostas 200 e 500. Para estimar o P95 apenas das respostas bem-sucedidas, filtre o seletor:

  ```promql
  histogram_quantile(
    0.95,
    sum by (le) (
      rate(bentoml_service_request_duration_seconds_bucket{http_response_code="200"}[5m])
    )
  )
  ```

#### Como atualizar o arquivo

Exemplo em PowerShell, com o serviço em execução (ajuste se o arquivo for gerado por outro método):

```powershell
curl.exe -s http://localhost:3000/metrics |
  Select-String 'endpoint="/predict"' |
  ForEach-Object { $_.Line } |
  Set-Content .\logs\metrics_predict.txt -Encoding utf8
```

## 4. Logs versus métricas

| Aspecto | Logs JSONL | Métricas BentoML |
| --- | --- | --- |
| Granularidade | Um evento por linha | Valores agregados |
| Detalhes de entrada e saída | Sim, conforme disponibilidade | Não representam os detalhes da inferência |
| Investigação por `request_id` | Sim | Não é o objetivo principal |
| Latência | Individual, em ms | Agregada, em segundos |
| Reinício do serviço | O arquivo continua se o disco for preservado | Contadores e histogramas em memória são reiniciados |
| Histórico | Até remoção ou arquivamento do arquivo | Depende do Prometheus ou de outro coletor |

O endpoint `/metrics` mostra o estado de runtime do processo. Para manter histórico e criar painéis, configure o Prometheus para coletar o endpoint periodicamente; o Grafana pode ser usado para visualização.

Exemplo mínimo de `prometheus.yml` (adapte ao seu ambiente):

```yaml
scrape_configs:
  - job_name: voltlens-ai
    scrape_interval: 15s
    static_configs:
      - targets: ["localhost:3000"]
```

## 5. Retenção dos registros

### 5.1 Comportamento implementado

Atualmente, os eventos são gravados em `logs/inference.log` por um `FileHandler` em modo append. Isso significa que:

- novas linhas são adicionadas ao arquivo existente;
- registros anteriores não são sobrescritos na gravação normal;
- o arquivo permanece após reiniciar o processo, desde que não seja removido e o armazenamento continue existindo;
- **não há rotação automática configurada**;
- **não há limite de tamanho nem prazo de expiração implementado**.

Portanto, a versão atual não define um período automático de retenção, e o arquivo cresce enquanto novas inferências forem registradas. Em containers com armazenamento efêmero, a destruição do container pode apagar os arquivos locais. Em produção, use volume persistente ou logging centralizado.

### 5.2 Política recomendada para produção

Como proposta de evolução — **não implementada nesta versão** — pode-se adotar:

| Etapa | Exemplo de política |
| --- | --- |
| Logs ativos | Últimos 30 dias |
| Arquivamento | Até 90 dias adicionais |
| Descarte | Exclusão ao fim do prazo definido |
| Rotação | Diária ou ao atingir um limite de tamanho |
| Segurança | Acesso restrito e minimização de dados sensíveis |

Os prazos devem ser definidos conforme requisitos operacionais, custo, auditoria e privacidade. A rotação pode ser implementada com um handler de rotação do Python (como `RotatingFileHandler` ou `TimedRotatingFileHandler`), com um mecanismo do sistema operacional (como `logrotate`) ou com um serviço centralizado.

### 5.3 Retenção das métricas

As métricas expostas pelo BentoML ficam na memória do processo e são reiniciadas quando o serviço reinicia. Para preservar o histórico, configure um Prometheus para coletar e armazenar as amostras. O prazo de retenção das métricas é definido no Prometheus e é independente da retenção dos arquivos JSONL.

## 6. Fluxo recomendado de análise

1. Inicie o serviço BentoML.
2. Envie requisições válidas e inválidas ao `POST /predict`.
3. Consulte `/metrics` e confirme os nomes e labels disponíveis.
4. Se quiser guardar o estado atual, atualize o snapshot `logs/metrics_predict.txt` (seção 3.5).
5. Use PromQL para acompanhar taxa de erro, média e P95 de latência.
6. Consulte `logs/inference.log` para investigar eventos individuais.
7. Execute o notebook para gerar estatísticas, histogramas, boxplots e o CSV tratado.
8. Compare períodos para identificar mudanças no volume, nas características das imagens, na confiança das detecções e no comportamento da API.

Logs e métricas se complementam: as métricas ajudam a identificar mudanças agregadas, enquanto os logs permitem descobrir quais requisições e entradas estão associadas a essas mudanças.
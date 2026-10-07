# VoltLens AI

API de inferência para detecção de medidores de energia elétrica em imagens, desenvolvida como parte da disciplina de MLOps.

O projeto transforma um modelo de visão computacional previamente treinado em um serviço HTTP utilizando **BentoML**, permitindo que outras aplicações enviem uma imagem e recebam como resposta a detecção do medidor, sua localização na imagem e a confiança da predição.

## Objetivo

O objetivo é disponibilizar o modelo de detecção de medidores como um serviço consumível por outras aplicações.

O fluxo da aplicação é:

```text
Imagem
   ↓
API BentoML
   ↓
Pré-processamento
   ↓
MeterNetwork
   ↓
Inferência
   ↓
JSON
```

O serviço recebe uma imagem de um medidor através do endpoint `/predict` e retorna:

* indicação se um medidor foi detectado;
* confiança da detecção;
* bounding box normalizada do objeto detectado.

---

## Modelo utilizado

O modelo utilizado é uma rede neural convolucional desenvolvida especificamente para a detecção de medidores.

A arquitetura é composta por camadas convolucionais, funções de ativação ReLU e operações de Max Pooling, finalizando em uma camada convolucional responsável por gerar as informações da detecção.

### - Entrada

A imagem é:

1. convertida para RGB;
2. redimensionada para `360 × 480`;
3. convertida para escala de cinza;
4. convertida em tensor;
5. enviada para o modelo.

### - Saída

O modelo produz um grid de predições contendo:

* posição `x`;
* posição `y`;
* largura;
* altura;
* confiança/objectness.

A melhor predição é selecionada e convertida para coordenadas normalizadas.

### Treinamento

O modelo foi treinado anteriormente como parte do projeto de visão computacional.

O checkpoint utilizado neste serviço contém os pesos treinados do `MeterNetwork`.

O serviço não realiza treinamento. Sua responsabilidade é carregar o modelo e disponibilizar a inferência por meio de uma API.

---

## Limitações atuais

O modelo atualmente realiza **detecção do medidor**.

Ele ainda não realiza:

* OCR do número do medidor;
* leitura do consumo;
* identificação automática de todas as funções do equipamento;
* comparação entre a leitura da imagem e uma leitura de referência;
* classificação de divergência.

Essas funcionalidades fazem parte de possíveis evoluções do projeto.

Além disso, a qualidade da detecção pode variar de acordo com fatores como:

* iluminação;
* posição da câmera;
* distância;
* enquadramento;
* oclusões;
* qualidade da imagem.

---

## Arquitetura do serviço

O serviço foi desenvolvido utilizando **BentoML**.

A estrutura principal é:

```text
voltlensAI/
├── model/
│   └── meter_detector.pt
├── src/
│   └── voltlensai/
│       ├── __init__.py
│       ├── model.py
│       ├── inference.py
│       └── service.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### Responsabilidade dos arquivos

**`model.py`**

Define a arquitetura da rede neural `MeterNetwork`.

**`inference.py`**

Responsável pelo carregamento do checkpoint, pré-processamento da imagem e execução da inferência.

**`service.py`**

Expõe o modelo como uma API utilizando BentoML.

**`model/meter_detector.pt`**

Checkpoint contendo os pesos treinados do modelo.

---

# Execução

## Requisitos

* Python 3.10 ou superior
* `uv`
* Git

A versão do Python utilizada no desenvolvimento deve ser compatível com a versão especificada no `pyproject.toml`.

---

## 1. Clonar o projeto

```bash
git clone https://github.com/FelipeSalustiano/voltlensAI.git
cd voltlensAI
```

## 2. Instalar as dependências

O projeto utiliza `uv` para gerenciamento do ambiente e das dependências.

```bash
uv sync
```

O arquivo `uv.lock` está versionado no repositório para garantir uma instalação reproduzível.

---

## 3. Iniciar o serviço

Execute:

```bash
uv run bentoml serve src/voltlensai/service.py:MeterDetectionService
```

Após a inicialização, o serviço estará disponível em:

```text
http://localhost:3000
```

---

# Testando pelo Swagger

O BentoML disponibiliza uma interface Swagger/OpenAPI para interação com o serviço.

Acesse:

```text
http://localhost:3000
```

Localize o endpoint:

```text
POST /predict
```

Clique em **Try it out**, selecione uma imagem e execute a requisição.

O serviço processará a imagem e retornará o resultado da detecção.

---

# Exemplo de resposta

Quando um medidor é detectado, a API retorna:

```json
{
  "detected": true,
  "confidence": 0.6720746159553528,
  "bounding_box": {
    "center_x": 0.8110498487949371,
    "center_y": 0.22479363481203715,
    "width": 0.7532642483711243,
    "height": 0.5677396059036255
  }
}
```

### Campos

| Campo        | Descrição                           |
| ------------ | ----------------------------------- |
| `detected`   | Indica se um medidor foi detectado  |
| `confidence` | Confiança da detecção               |
| `center_x`   | Coordenada X central normalizada    |
| `center_y`   | Coordenada Y central normalizada    |
| `width`      | Largura da bounding box normalizada |
| `height`     | Altura da bounding box normalizada  |

As coordenadas da bounding box são normalizadas entre `0` e `1`, para o modelo não ficar preso a uma resolução específica.

---

# Contrato da API

### Endpoint

```text
POST /predict
```

### Entrada

Imagem enviada diretamente no corpo da requisição.

Exemplo:

```text
Content-Type: image/jpeg
```

### Saída

Objeto JSON contendo:

```json
{
  "detected": true,
  "confidence": 0.67,
  "bounding_box": {
    "center_x": 0.81,
    "center_y": 0.22,
    "width": 0.75,
    "height": 0.57
  }
}
```

---

# Possíveis evoluções

Como próximos passos, o serviço pode evoluir para um pipeline mais completo de leitura de medidores:

```text
Imagem
   ↓
Detecção do medidor
   ↓
Recorte da região
   ↓
OCR
   ↓
Leitura dos valores
   ↓
Validação com dados de referência
   ↓
Classificação
```

Entre as possíveis melhorias estão:

* implementação de OCR;
* identificação do número do medidor;
* leitura do consumo;
* tratamento de diferentes condições de iluminação;
* aumento e diversificação do dataset;
* avaliação com métricas específicas de detecção;
* monitoramento das inferências;
* versionamento dos modelos.

---

# Uso de Inteligência Artificial

Ferramentas de Inteligência Artificial generativa foram utilizadas como apoio durante o desenvolvimento.

A IA foi utilizada principalmente para:

* auxílio na estruturação do serviço;
* esclarecimento de conceitos e APIs;
* revisão de código;
* identificação e correção de problemas de integração;
* apoio na elaboração da documentação.

Todo o código utilizado no projeto foi revisado e validado durante a execução local do serviço.

A utilização de IA não substituiu a compreensão e validação do funcionamento do código.

---

# Licença

Este projeto está disponibilizado sob a licença MIT.

Consulte o arquivo [`LICENSE`](LICENSE) para mais informações.

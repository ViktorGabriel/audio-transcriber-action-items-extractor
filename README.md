---
title: Audio Transcriber & Action Items Extractor
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
short_description: Transcrição de áudio e extração de itens de ação
---

# 🎙️ Audio Transcriber & Action Items Extractor

Aplicação corporativa modular projetada para execução no **Hugging Face Spaces** em ambiente gratuito com **CPU (2 vCPU, 16 GB RAM)**. Transforma gravações de reuniões, palestras e conversas em transcrições fiéis e extrai automaticamente listas estruturadas de itens de ação (*action items*), responsáveis, decisões e prazos.

---

## 🏗️ Arquitetura do Projeto

A solução adota uma arquitetura em camadas orientada a serviços e contratos de dados tipados:

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml             # Pipeline de CI (Lint + Pytest)
├── src/
│   ├── __init__.py
│   ├── schemas.py             # Modelos de dados Pydantic (contratos de entrada/saída)
│   └── services/
│       ├── __init__.py
│       ├── transcriber.py     # Serviço de transcrição de áudio (faster-whisper)
│       └── extractor.py       # Serviço de extração de itens de ação e sumário (LLM)
├── tests/
│   ├── test_schemas.py        # Testes de validação dos contratos Pydantic
│   └── test_services.py       # Testes de integração/stub dos serviços
├── .env.example               # Template de variáveis de ambiente
├── .gitignore                 # Higiene rigorosa de repositório e zero-leak
├── app.py                     # Entrypoint da interface web Gradio
├── packages.txt               # Dependências de sistema do Space (ffmpeg)
├── requirements.txt           # Dependências Python otimizadas para CPU
└── README.md                  # Metadados do Space e documentação técnica
```

---

## ⚡ Otimização para CPU Gratuita (2 vCPU / 16 GB RAM)

1. **Transcrição Eficiente (`faster-whisper`):**
   - Utilização do CTranslate2 com quantização `int8` para máxima performance em CPU com baixo footprint de memória.
2. **PyTorch CPU-only:**
   - Pacotes compilados exclusivamente para CPU (`whl/cpu`) eliminando o overhead de 2GB+ de bibliotecas CUDA desnecessárias.
3. **Contratos Fortemente Tipados (`pydantic`):**
   - Validação estrita de schemas em tempo de execução garantindo parsing seguro das saídas.

---

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
- Python 3.12+
- `ffmpeg` instalado no sistema operacional

### 2. Instalação

```bash
# Crie e ative um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Execução

```bash
python app.py
```
Acesse a aplicação no navegador em `http://localhost:7860`.

---

## 🧪 Testes Automatizados

```bash
pytest -v
```

---

## 📄 Licença

Distribuído sob a licença MIT.

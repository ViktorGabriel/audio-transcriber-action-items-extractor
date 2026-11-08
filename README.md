---
title: Audio Transcriber & Action Items Extractor
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
short_description: Transcrição de áudio e extração inteligente de itens de ação
---

# 🎙️ Audio Transcriber & Action Items Extractor

Aplicação corporativa desenvolvida para **Hugging Face Spaces** com foco em transcrição de reuniões em áudio e extração automatizada de itens de ação (*action items*), responsáveis e prazos.

---

## 🚀 Arquitetura & Tecnologias

- **Interface:** [Gradio](https://www.gradio.app/)
- **Modelos de Transcrição:** OpenAI Whisper (`openai/whisper-large-v3-turbo`)
- **Processamento de Áudio:** `ffmpeg`
- **Extração Semântica:** Hugging Face Transformers & Pipelines de Linguagem
- **CI/CD:** GitHub Actions com testes automatizados e linters
- **Ambiente:** Hugging Face Spaces (Python 3.12)

---

## 📂 Estrutura do Projeto

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml          # Pipeline de integração contínua (Lint + Testes)
├── tests/
│   └── test_app.py         # Suíte de testes unitários
├── .env.example            # Template de variáveis de ambiente
├── .gitignore              # Regras de exclusão e higiene de repositório
├── app.py                  # Ponto de entrada da aplicação Gradio
├── packages.txt            # Dependências de sistema no Hugging Face (ffmpeg)
├── requirements.txt        # Dependências Python gerenciadas
└── README.md               # Metadados do Space e documentação técnica
```

---

## 🛠️ Como Executar Localmente

### 1. Pré-requisitos
- Python 3.12+
- `ffmpeg` instalado no sistema operacional

### 2. Instalação

```bash
# Crie e ative um ambiente virtual
python -m venv .venv
# No Linux/macOS:
# source .venv/bin/activate
# No Windows:
# .venv\Scripts\Activate.ps1

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configuração de Variáveis de Ambiente

Copie o arquivo de exemplo e preencha suas chaves caso necessário:
```bash
cp .env.example .env
```

### 4. Executando a Aplicação

```bash
python app.py
```
Acesse a aplicação no navegador em `http://localhost:7860`.

---

## 🧪 Testes Automatizados

Para executar os testes unitários:

```bash
pytest -v
```

---

## 📄 Licença

Distribuído sob a licença MIT. Consulte o arquivo correspondente para mais informações.

---
title: Audio Transcriber & Action Items Extractor
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
license: mit
short_description: Transcrição de áudio e extração inteligente de itens de ação via Gradio-Lite
---

# 🎙️ Audio Transcriber & Action Items Extractor (Gradio-Lite)

Aplicação corporativa desenvolvida para **Hugging Face Spaces (Static)** utilizando **Gradio-Lite** (WebAssembly / Pyodide). Executa diretamente no navegador sem custos de servidor de backend.

---

## 🚀 Arquitetura & Tecnologias

- **Ambiente Space:** `sdk: static` (Hospedagem estática gratuita do Hugging Face)
- **Frontend / Engine:** [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (`@gradio/lite` via WebAssembly & Pyodide)
- **Execução:** 100% Client-side no navegador (ou chamadas via Hugging Face Serverless Inference API / Web APIs)
- **CI/CD:** GitHub Actions com testes automatizados e validações de integridade

---

## 📂 Estrutura do Projeto

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml          # Pipeline de integração contínua
├── tests/
│   └── test_app.py         # Testes de integridade da interface e lógica
├── .env.example            # Template de variáveis de ambiente
├── .gitignore              # Regras de exclusão e higiene de repositório
├── index.html              # Ponto de entrada estático com Gradio-Lite
├── app.py                  # Lógica Python compatível com Pyodide / Gradio
└── README.md               # Metadados do Space (sdk: static) e documentação técnica
```

---

## 🛠️ Como Executar Localmente

### Opção 1: Servidor HTTP Local (Navegador)

Como o projeto é estático com WebAssembly, você pode rodar qualquer servidor HTTP simples:

```bash
# Com Python nativo:
python -m http.server 7860
```
Acesse `http://localhost:7860` no navegador.

### Opção 2: Python / Gradio Tradicional

```bash
pip install -r requirements.txt
python app.py
```

---

## 🧪 Testes Automatizados

```bash
pytest -v
```

---

## 📄 Licença

Distribuído sob a licença MIT.

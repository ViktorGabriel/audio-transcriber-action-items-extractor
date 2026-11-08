"""
Audio Transcriber & Action Items Extractor
Ponto de entrada da aplicação para Hugging Face Spaces (Gradio).
"""

import os
from typing import Tuple
import gradio as gr
from dotenv import load_dotenv

from src.services.transcriber import AudioTranscriber
from src.services.extractor import ActionItemExtractor

load_dotenv()

# Instanciação dos serviços
transcriber_service = AudioTranscriber(model_size="base")
extractor_service = ActionItemExtractor()


def process_audio(audio_path: str | None) -> Tuple[str, str, str]:
    """
    Função controladora que orquestra a transcrição e extração de ações.

    Args:
        audio_path: Caminho do arquivo de áudio carregado pelo usuário.

    Returns:
        Tupla com (transcrição bruta, resumo da reunião, itens de ação formatados).
    """
    if not audio_path or not os.path.exists(audio_path):
        return (
            "Nenhum arquivo de áudio válido foi fornecido.",
            "Sem dados para gerar resumo.",
            "Nenhum item de ação detectado.",
        )

    # Execução via serviços estruturados (stubs da Etapa 1)
    transcription = transcriber_service.transcribe(audio_path=audio_path)
    extraction = extractor_service.extract(transcription_text=transcription.raw_text)

    # Formatação das ações para Markdown
    action_items_md = "### ✅ Itens de Ação Detectados:\n\n"
    for item in extraction.action_items:
        action_items_md += (
            f"- **[{item.priority.value.upper()}]** {item.task}\n"
            f"  - **Responsável:** {item.assignee or 'Não atribuído'}\n"
            f"  - **Prazo:** {item.deadline or 'Não informado'}\n\n"
        )

    return transcription.raw_text, extraction.summary, action_items_md


def create_demo() -> gr.Blocks:
    """Constrói o layout visual em Gradio."""
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="emerald",
    )

    with gr.Blocks(theme=custom_theme, title="Audio Transcriber & Action Items") as demo:
        gr.Markdown(
            """
            # 🎙️ Audio Transcriber & Action Items Extractor
            Transforme gravações de reuniões e palestras em **transcrições precisas** e **itens de ação estruturados**.
            *Ambiente otimizado para Hugging Face Spaces CPU (2 vCPU, 16 GB RAM).*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Entrada de Áudio (WAV, MP3, M4A)",
                )
                process_btn = gr.Button("Processar Áudio", variant="primary")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Transcrição"):
                        transcription_output = gr.Textbox(
                            label="Texto Transcrito",
                            lines=10,
                            placeholder="A transcrição aparecerá aqui...",
                        )
                    with gr.TabItem("Resumo & Ações"):
                        summary_output = gr.Textbox(
                            label="Resumo Executivo",
                            lines=3,
                            placeholder="O resumo da reunião aparecerá aqui...",
                        )
                        actions_output = gr.Markdown(
                            value="*Aguardando processamento...*",
                        )

        process_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[transcription_output, summary_output, actions_output],
        )

    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

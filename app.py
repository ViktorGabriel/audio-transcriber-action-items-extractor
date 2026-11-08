"""
Audio Transcriber & Action Items Extractor
Aplicação para Hugging Face Spaces com Gradio.
"""

import os
from typing import Tuple
import gradio as gr
from dotenv import load_dotenv

load_dotenv()


def process_audio(audio_path: str | None) -> Tuple[str, str]:
    """
    Processa o arquivo de áudio enviado:
    - Realiza a transcrição via Whisper
    - Extrai itens de ação e sumário de tarefas
    """
    if not audio_path:
        return "Nenhum arquivo de áudio fornecido.", "Nenhuma ação detectada."

    # Placeholder estrutural para implementação completa no próximo prompt
    transcription = f"Arquivo recebido com sucesso: {os.path.basename(audio_path)}"
    action_items = (
        "- [ ] Definir pipeline de transcrição (Whisper)\n"
        "- [ ] Configurar modelo de extração de Action Items\n"
        "- [ ] Validar formato de saída e exportação"
    )
    return transcription, action_items


def build_interface() -> gr.Blocks:
    """Constrói a interface Gradio para Hugging Face Spaces."""
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )

    with gr.Blocks(theme=custom_theme, title="Audio Transcriber & Action Items") as demo:
        gr.Markdown(
            """
            # 🎙️ Audio Transcriber & Action Items Extractor
            Transforme gravações de reuniões e áudios em **transcrições precisas** e **itens de ação estruturados**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Entrada de Áudio",
                )
                submit_btn = gr.Button("Processar Áudio", variant="primary")

            with gr.Column(scale=1):
                transcription_output = gr.Textbox(
                    label="Transcrição Completa",
                    lines=8,
                    placeholder="A transcrição aparecerá aqui...",
                )
                action_items_output = gr.Markdown(
                    label="Itens de Ação (Action Items)",
                    value="*Aguardando processamento...*",
                )

        submit_btn.click(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[transcription_output, action_items_output],
        )

    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

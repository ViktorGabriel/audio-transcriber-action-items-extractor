"""
Audio Transcriber & Action Items Extractor
Aplicação corporativa para Hugging Face Spaces (Gradio Blocks).
"""

import logging
import os
from typing import Any, Dict, List, Tuple
import gradio as gr
from dotenv import load_dotenv

from src.schemas import ExtractionResult
from src.services.extractor import ActionItemExtractor
from src.services.transcriber import AudioTranscriber

load_dotenv()

# Configuração de logging
logger = logging.getLogger("app")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    )

# Inicialização dos serviços (Singleton / Lazy Loading)
transcriber = AudioTranscriber(model_size="base", device="cpu", compute_type="int8")
extractor = ActionItemExtractor()


def format_action_items_dataframe(
    extraction: ExtractionResult,
) -> List[List[str]]:
    """
    Converte a lista de ActionItem do Pydantic para uma matriz de linhas para o gr.Dataframe.

    Args:
        extraction: Resultado estruturado da extração semântica.

    Returns:
        List[List[str]]: Matriz bidimensional com Tarefa, Responsável, Prioridade e Prazo.
    """
    rows: List[List[str]] = []
    for item in extraction.action_items:
        rows.append(
            [
                item.task,
                item.assignee or "Não atribuído",
                (item.priority or "media").capitalize(),
                item.deadline_hint or "Não informado",
            ]
        )
    return rows


def process_audio_pipeline(
    audio_path: str | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Tuple[str, str, List[List[str]], Dict[str, Any], str]:
    """
    Pipeline orquestrador conectado aos eventos do Gradio.
    Executa a transcrição do áudio e extrai o sumário executivo e itens de ação.

    Args:
        audio_path: Caminho do arquivo temporário gerado pelo Gradio.
        progress: Instância do gr.Progress para feedback visual em tempo real.

    Returns:
        Tupla com:
            - Resumo Executivo (str)
            - Transcrição Completa (str)
            - Matriz de Itens de Ação para o gr.Dataframe (list)
            - Dicionário JSON bruto (dict)
            - Badge informativa de status/idioma (str)
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise gr.Error("Nenhum arquivo de áudio foi fornecido. Envie um arquivo ou grave pelo microfone.")

    try:
        # Etapa 1: Transcrição
        progress(0.15, desc="Transcrevendo áudio com faster-whisper...")
        logger.info("Iniciando transcrição para o arquivo: %s", audio_path)
        transcript_text, detected_lang = transcriber.transcribe(audio_path=audio_path)

        if not transcript_text.strip():
            logger.warning("Transcrição retornou texto vazio para: %s", audio_path)
            return (
                "Nenhuma fala compreensível foi detectada no áudio.",
                "Áudio silencioso ou sem conteúdo de voz perceptível.",
                [],
                {"status": "empty_audio", "language": detected_lang},
                f"⚠️ Idioma detectado: **{detected_lang.upper()}** (sem fala)",
            )

        # Etapa 2: Extração Semântica
        progress(0.65, desc="Extraindo tarefas e resumo executivo...")
        logger.info(
            "Extraindo ações da transcrição (idioma: %s, caracteres: %d)...",
            detected_lang,
            len(transcript_text),
        )
        extraction = extractor.extract_actions(
            transcript=transcript_text,
            language=detected_lang,
        )

        # Etapa 3: Formatação e Conclusão
        progress(0.90, desc="Formatando resultados estruturados...")
        dataframe_rows = format_action_items_dataframe(extraction)
        json_payload = extraction.model_dump()
        json_payload["raw_transcription"] = transcript_text

        status_badge = (
            f"✅ **Processado com sucesso!** | Idioma: `{detected_lang.upper()}` | "
            f"Tarefas identificadas: `{len(extraction.action_items)}`"
        )

        progress(1.0, desc="Concluído!")
        logger.info("Pipeline finalizado com sucesso.")

        return (
            extraction.summary,
            transcript_text,
            dataframe_rows,
            json_payload,
            status_badge,
        )

    except Exception as exc:
        logger.error("Erro crítico na execução do pipeline: %s", exc, exc_info=True)
        raise gr.Error(f"Erro ao processar áudio: {str(exc)}") from exc


def create_demo() -> gr.Blocks:
    """Constrói a interface completa com Gradio Blocks."""
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, title="Audio Transcriber & Action Items Extractor") as demo:
        # Header / Badges
        gr.Markdown(
            """
            # 🎙️ Audio Transcriber & Action Items Extractor
            Transforme reuniões, aulas e conversas em **transcrições fiéis**, **resumos executivos**
            e **listas estruturadas de tarefas** com responsáveis e prazos.

            ---
            `Ambiente: Hugging Face Spaces (CPU 2 vCPU • 16 GB)` | `ASR: faster-whisper` | `LLM: Qwen2.5-1.5B`
            """
        )

        with gr.Row():
            # Coluna Esquerda: Entrada de Áudio e Ação
            with gr.Column(scale=5):
                gr.Markdown("### 1. Entrada de Áudio")
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Faça upload ou grave pelo microfone",
                )

                submit_btn = gr.Button(
                    "Transcrever e Extrair Tarefas",
                    variant="primary",
                    size="lg",
                )

                status_output = gr.Markdown(
                    value="*Aguardando áudio para iniciar o processamento...*",
                )

                gr.Examples(
                    examples=[],
                    inputs=audio_input,
                    label="Exemplos de Demonstração",
                )

            # Coluna Direita: Abas de Resultados
            with gr.Column(scale=7):
                gr.Markdown("### 2. Resultados da Análise")
                with gr.Tabs():
                    # Aba 1: Resumo e Transcrição
                    with gr.TabItem("📋 Resumo & Transcrição", id="tab_summary"):
                        gr.Markdown("#### Resumo Executivo")
                        summary_output = gr.Textbox(
                            label="Resumo Conciso da Reunião",
                            lines=4,
                            placeholder="O resumo executivo será gerado aqui...",
                            interactive=False,
                        )

                        gr.Markdown("#### Transcrição Completa")
                        transcript_output = gr.Textbox(
                            label="Texto Integral Transcrito",
                            lines=9,
                            placeholder="A transcrição integral aparecerá aqui...",
                            interactive=False,
                        )

                    # Aba 2: Tabela de Itens de Ação
                    with gr.TabItem("✅ Itens de Ação", id="tab_actions"):
                        gr.Markdown("#### Tarefas e Ações Identificadas")
                        dataframe_output = gr.Dataframe(
                            headers=["Tarefa", "Responsável", "Prioridade", "Prazo Estimado"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                        )

                    # Aba 3: JSON Bruto
                    with gr.TabItem("🔍 JSON Estruturado", id="tab_json"):
                        gr.Markdown("#### Exportação / Cópia de Dados Estruturados")
                        json_output = gr.JSON(
                            label="Payload Completo da Análise",
                        )

        # Vinculação do Evento de Clique
        submit_btn.click(
            fn=process_audio_pipeline,
            inputs=[audio_input],
            outputs=[
                summary_output,
                transcript_output,
                dataframe_output,
                json_output,
                status_output,
            ],
        )

    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

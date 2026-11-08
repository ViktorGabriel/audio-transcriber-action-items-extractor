"""
Testes unitários da interface e pipeline em app.py.
"""

from unittest.mock import MagicMock
import pytest
import gradio as gr

from app import (
    create_demo,
    format_action_items_dataframe,
    process_audio_pipeline,
)
from src.schemas import ActionItem, ExtractionResult


def test_create_demo_instantiation():
    demo = create_demo()
    assert demo is not None
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Audio Transcriber & Action Items Extractor"


def test_format_action_items_dataframe():
    extraction = ExtractionResult(
        summary="Reunião de teste.",
        action_items=[
            ActionItem(
                task="Deploy no Space",
                assignee="Viktor",
                priority="alta",
                deadline_hint="Hoje",
            ),
            ActionItem(
                task="Revisar código",
                assignee=None,
                priority="media",
                deadline_hint=None,
            ),
        ],
        language="pt",
    )
    rows = format_action_items_dataframe(extraction)
    assert len(rows) == 2
    assert rows[0] == ["Deploy no Space", "Viktor", "Alta", "Hoje"]
    assert rows[1] == ["Revisar código", "Não atribuído", "Media", "Não informado"]


def test_process_audio_pipeline_missing_audio():
    with pytest.raises(gr.Error, match="Nenhum arquivo de áudio foi fornecido"):
        process_audio_pipeline(None)


def test_process_audio_pipeline_success(monkeypatch):
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = (
        "Reunião de alinhamento com a equipe.",
        "pt",
    )

    mock_extractor = MagicMock()
    mock_extractor.extract_actions.return_value = ExtractionResult(
        summary="Resumo da reunião executiva.",
        action_items=[
            ActionItem(
                task="Entregar relatório final",
                assignee="Carlos",
                priority="alta",
                deadline_hint="Sexta-feira",
            )
        ],
        language="pt",
        key_decisions=["Release aprovada."],
    )

    monkeypatch.setattr("app.transcriber", mock_transcriber)
    monkeypatch.setattr("app.extractor", mock_extractor)

    # Cria mock de caminho de arquivo existente
    monkeypatch.setattr("os.path.isfile", lambda path: True)

    dummy_progress = MagicMock()
    summary, transcript, rows, json_data, status_badge = process_audio_pipeline(
        "dummy_meeting.wav",
        progress=dummy_progress,
    )

    assert summary == "Resumo da reunião executiva."
    assert transcript == "Reunião de alinhamento com a equipe."
    assert len(rows) == 1
    assert rows[0][0] == "Entregar relatório final"
    assert json_data["language"] == "pt"
    assert "✅ **Processado com sucesso!**" in status_badge

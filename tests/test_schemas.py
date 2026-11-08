"""
Testes unitários dos contratos de dados (Pydantic schemas).
"""

from src.schemas import (
    ActionItem,
    PriorityLevel,
    TranscriptionSegment,
    TranscriptionResult,
    ExtractionResult,
    AnalysisResponse,
)


def test_action_item_schema():
    item = ActionItem(
        task="Configurar repositório Hugging Face",
        assignee="Viktor",
        priority=PriorityLevel.HIGH,
        deadline="Hoje",
    )
    assert item.task == "Configurar repositório Hugging Face"
    assert item.priority == PriorityLevel.HIGH
    assert item.assignee == "Viktor"


def test_transcription_result_schema():
    segment = TranscriptionSegment(id=1, start=0.0, end=2.5, text="Olá mundo")
    result = TranscriptionResult(
        raw_text="Olá mundo",
        language="pt",
        duration_seconds=2.5,
        segments=[segment],
    )
    assert result.duration_seconds == 2.5
    assert len(result.segments) == 1
    assert result.segments[0].text == "Olá mundo"


def test_analysis_response_schema():
    transcription = TranscriptionResult(
        raw_text="Alinhamento sobre prazos.",
        language="pt",
        duration_seconds=5.0,
        segments=[],
    )
    extraction = ExtractionResult(
        summary="Reunião de alinhamento.",
        action_items=[
            ActionItem(task="Enviar relatório", priority=PriorityLevel.MEDIUM)
        ],
        key_decisions=["Decidido o prazo final."],
    )
    response = AnalysisResponse(
        transcription=transcription,
        extraction=extraction,
    )
    assert response.transcription.raw_text == "Alinhamento sobre prazos."
    assert len(response.extraction.action_items) == 1
    assert len(response.extraction.key_decisions) == 1

"""
Testes unitários dos contratos de dados Pydantic (src/schemas.py).
"""

from src.schemas import ActionItem, ExtractionResult


def test_action_item_schema():
    item = ActionItem(
        task="Definir arquitetura do pipeline",
        assignee="Viktor",
        priority="alta",
        deadline_hint="Até o final da semana",
    )
    assert item.task == "Definir arquitetura do pipeline"
    assert item.assignee == "Viktor"
    assert item.priority == "alta"
    assert item.deadline_hint == "Até o final da semana"
    assert item.deadline == "Até o final da semana"


def test_action_item_defaults():
    item = ActionItem(task="Tarefa simples")
    assert item.task == "Tarefa simples"
    assert item.assignee is None
    assert item.priority == "media"
    assert item.deadline_hint is None


def test_extraction_result_schema():
    action = ActionItem(
        task="Configurar secrets no Hugging Face",
        assignee="DevOps",
        priority="alta",
        deadline_hint="Hoje",
    )
    result = ExtractionResult(
        summary="Alinhamento sobre deploy no Hugging Face Spaces.",
        action_items=[action],
        language="pt",
        key_decisions=["Uso do Whisper base com quantização int8."],
    )
    assert "deploy" in result.summary
    assert len(result.action_items) == 1
    assert result.action_items[0].assignee == "DevOps"
    assert result.language == "pt"
    assert len(result.key_decisions) == 1

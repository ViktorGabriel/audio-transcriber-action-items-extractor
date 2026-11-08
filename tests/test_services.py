"""
Testes unitários dos serviços de transcrição e extração semântica.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock
import pytest

from src.schemas import ExtractionResult
from src.services.transcriber import AudioTranscriber
from src.services.extractor import (
    ActionItemExtractor,
    extract_actions,
    sanitize_json_response,
)


def test_transcriber_singleton_pattern():
    instance1 = AudioTranscriber()
    instance2 = AudioTranscriber()
    assert instance1 is instance2


def test_transcriber_validate_empty_path():
    transcriber = AudioTranscriber()
    with pytest.raises(ValueError, match="string não vazia"):
        transcriber.validate_audio_file("")


def test_transcriber_validate_file_not_found():
    transcriber = AudioTranscriber()
    with pytest.raises(FileNotFoundError, match="não encontrado"):
        transcriber.validate_audio_file("caminho_totalmente_inexistente.wav")


def test_transcriber_validate_empty_file():
    transcriber = AudioTranscriber()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="0 bytes"):
            transcriber.validate_audio_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_transcriber_validate_unsupported_format():
    transcriber = AudioTranscriber()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"conteudo de texto")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="não suportado"):
            transcriber.validate_audio_file(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_transcriber_transcribe_mocked():
    transcriber = AudioTranscriber()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF dummy audio content")
        temp_path = f.name

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = " Transcrição de teste concluída. "
    mock_info = MagicMock()
    mock_info.language = "pt"

    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    transcriber._model = mock_model

    try:
        text, lang = transcriber.transcribe(temp_path)
        assert text == "Transcrição de teste concluída."
        assert lang == "pt"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =========================================================================
# Testes do Extrator de Ações e Sanitização
# =========================================================================

def test_sanitize_json_response_clean():
    sample = '{"summary": "Teste", "action_items": []}'
    assert sanitize_json_response(sample) == sample


def test_sanitize_json_response_markdown_blocks():
    sample = '```json\n{"summary": "Teste", "action_items": []}\n```'
    assert sanitize_json_response(sample) == '{"summary": "Teste", "action_items": []}'


def test_sanitize_json_response_conversational():
    sample = (
        'Claro! Aqui está o JSON:\n'
        '```json\n{"summary": "Reunião de alinhamento", "action_items": []}\n```\n'
        'Espero ter ajudado!'
    )
    expected = '{"summary": "Reunião de alinhamento", "action_items": []}'
    assert sanitize_json_response(sample) == expected


def test_sanitize_json_response_empty():
    assert sanitize_json_response("") == "{}"
    assert sanitize_json_response(None) == "{}"


def test_extractor_empty_text():
    extractor = ActionItemExtractor()
    with pytest.raises(ValueError, match="não pode ser vazio"):
        extractor.extract_actions("")


def test_extractor_with_mocked_inference_client():
    extractor = ActionItemExtractor(hf_token="fake_token")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "summary": "Resumo de alinhamento de sprint.",
        "action_items": [
            {
                "task": "Atualizar dependências de segurança",
                "assignee": "Viktor",
                "priority": "alta",
                "deadline_hint": "Amanhã",
            }
        ],
        "language": "pt",
        "key_decisions": ["Aprovado o uso do Qwen2.5 para extração."],
    })
    mock_response.choices = [mock_choice]
    mock_client.chat_completion.return_value = mock_response

    extractor._inference_client = mock_client

    result = extractor.extract_actions("O Viktor precisa atualizar as dependências até amanhã.")
    assert "sprint" in result.summary
    assert len(result.action_items) == 1
    assert result.action_items[0].assignee == "Viktor"
    assert result.action_items[0].priority == "alta"
    assert result.language == "pt"


def test_extractor_json_error_fallback():
    extractor = ActionItemExtractor(hf_token="fake_token")

    mock_client = MagicMock()
    mock_choice = MagicMock()
    # Retorna texto não JSON
    mock_choice.message.content = "Desculpe, não consegui estruturar em JSON."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat_completion.return_value = mock_response

    extractor._inference_client = mock_client

    result = extractor.extract_actions("Discussão aberta sobre deploy.")
    assert "Sumário preliminar:" in result.summary
    assert result.action_items == []
    assert result.language == "pt"


def test_extract_actions_function_wrapper(monkeypatch):
    mock_extractor = MagicMock()
    mock_extractor.extract_actions.return_value = ExtractionResult(
        summary="Resumo de teste via wrapper",
        action_items=[],
        language="pt",
        key_decisions=["Decisão de teste"],
    )
    monkeypatch.setattr(
        "src.services.extractor.ActionItemExtractor.extract_actions",
        mock_extractor.extract_actions,
    )
    result = extract_actions("Transcrição rápida para teste.")
    assert result is not None
    assert result.summary == "Resumo de teste via wrapper"

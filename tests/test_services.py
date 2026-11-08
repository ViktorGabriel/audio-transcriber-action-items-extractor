"""
Testes unitários dos serviços de transcrição e extração.
"""

import os
import tempfile
from unittest.mock import MagicMock
import pytest

from src.services.transcriber import AudioTranscriber
from src.services.extractor import ActionItemExtractor


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

    # Cria arquivo temporário de áudio válido com bytes simulados
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF dummy audio content")
        temp_path = f.name

    # Mock do WhisperModel
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


def test_extractor_empty_text():
    extractor = ActionItemExtractor()
    with pytest.raises(ValueError, match="não pode ser vazio"):
        extractor.extract("")


def test_extractor_success():
    extractor = ActionItemExtractor()
    result = extractor.extract("Texto transcrito de teste para extração de tarefas.")
    assert result.summary is not None
    assert len(result.action_items) > 0
    assert result.language == "pt"
    assert result.action_items[0].deadline_hint is not None

"""
Testes unitários dos stubs de serviços.
"""

import pytest
import tempfile
import os
from src.services.transcriber import AudioTranscriber
from src.services.extractor import ActionItemExtractor


def test_transcriber_file_not_found():
    transcriber = AudioTranscriber()
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("caminho_inexistente.wav")


def test_transcriber_success_with_dummy_file():
    transcriber = AudioTranscriber()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        result = transcriber.transcribe(temp_path)
        assert result.raw_text is not None
        assert result.language == "pt"
        assert len(result.segments) > 0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_extractor_empty_text():
    extractor = ActionItemExtractor()
    with pytest.raises(ValueError):
        extractor.extract("")


def test_extractor_success():
    extractor = ActionItemExtractor()
    result = extractor.extract("Transcrição com itens de ação pendentes.")
    assert result.summary is not None
    assert len(result.action_items) > 0
    assert len(result.key_decisions) > 0

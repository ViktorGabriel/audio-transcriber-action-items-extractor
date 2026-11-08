"""
Testes unitários para o pipeline base do Audio Transcriber & Action Items Extractor.
"""

from app import process_audio, build_interface


def test_process_audio_empty():
    transcription, action_items = process_audio(None)
    assert transcription == "Nenhum arquivo de áudio fornecido."
    assert action_items == "Nenhuma ação detectada."


def test_process_audio_dummy_file():
    transcription, action_items = process_audio("/tmp/meeting_sample.wav")
    assert "meeting_sample.wav" in transcription
    assert "- [ ]" in action_items


def test_build_interface_instantiation():
    demo = build_interface()
    assert demo is not None

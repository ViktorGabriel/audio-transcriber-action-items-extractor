"""
Testes unitários da interface e orquestrador em app.py.
"""

from app import process_audio, create_demo


def test_process_audio_empty():
    transcription, summary, actions = process_audio(None)
    assert "Nenhum arquivo de áudio válido" in transcription
    assert "Sem dados para gerar resumo" in summary
    assert "Nenhum item de ação detectado" in actions


def test_create_demo_instantiation():
    demo = create_demo()
    assert demo is not None

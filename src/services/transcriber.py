"""
Serviço de transcrição de áudio baseado no faster-whisper.
Otimizado para execução em CPU com quantização int8 em ambientes Hugging Face Spaces.
"""

import os
import threading
from typing import Optional, Set, Tuple
from faster_whisper import WhisperModel


class AudioTranscriber:
    """
    Serviço singleton para transcrição de áudio utilizando faster-whisper.

    Garante que o modelo Whisper seja carregado apenas uma vez em memória (lazy loading),
    executando em CPU com quantização int8 para otimização de velocidade e memória RAM.
    """

    _instance: Optional["AudioTranscriber"] = None
    _lock: threading.Lock = threading.Lock()
    _model: Optional[WhisperModel] = None

    # Formatos de áudio suportados comumente pelo ffmpeg/faster-whisper
    SUPPORTED_EXTENSIONS: Set[str] = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac",
        ".aac",
        ".wma",
        ".webm",
        ".mp4",
    }

    def __new__(cls, *args, **kwargs) -> "AudioTranscriber":
        """Implementação thread-safe do padrão Singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """
        Inicializa os parâmetros do transcritor.

        Args:
            model_size: Nome do checkpoint Whisper (default: 'base').
            device: Dispositivo de execução ('cpu').
            compute_type: Quantização computacional ('int8').
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def _get_model(self) -> WhisperModel:
        """
        Obtém a instância carregada do modelo Whisper via Lazy Loading.
        Inicializa o modelo apenas na primeira chamada.
        """
        if self._model is not None:
            return self._model

        if AudioTranscriber._model is None:
            with AudioTranscriber._lock:
                if AudioTranscriber._model is None:
                    AudioTranscriber._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
        return AudioTranscriber._model

    def validate_audio_file(self, audio_path: str) -> None:
        """
        Valida a existência, integridade e extensão do arquivo de áudio.

        Args:
            audio_path: Caminho do arquivo a ser validado.

        Raises:
            ValueError: Se o caminho for nulo, vazio ou formato incompatível.
            FileNotFoundError: Se o arquivo não existir no disco.
        """
        if not audio_path or not isinstance(audio_path, str) or not audio_path.strip():
            raise ValueError("O caminho do arquivo de áudio deve ser uma string não vazia.")

        clean_path = audio_path.strip()

        if not os.path.isfile(clean_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: '{clean_path}'")

        if os.path.getsize(clean_path) == 0:
            raise ValueError(f"O arquivo de áudio '{clean_path}' está vazio (0 bytes).")

        _, ext = os.path.splitext(clean_path.lower())
        if ext not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Formato de áudio '{ext}' não suportado. "
                f"Formatos aceitos: {supported}"
            )

    def transcribe(
        self,
        audio_path: str,
        beam_size: int = 5,
    ) -> Tuple[str, str]:
        """
        Transcreve o arquivo de áudio fornecido utilizando faster-whisper.

        Args:
            audio_path: Caminho no disco para o arquivo de áudio.
            beam_size: Tamanho do feixe para decodificação (default: 5).

        Returns:
            Tuple[str, str]: (texto_transcrito, idioma_detectado).

        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
            ValueError: Se o arquivo for inválido ou de formato não suportado.
            RuntimeError: Se houver falha durante o processo de inferência.
        """
        self.validate_audio_file(audio_path)
        clean_path = audio_path.strip()

        try:
            model = self._get_model()
            segments, info = model.transcribe(
                clean_path,
                beam_size=beam_size,
            )

            # Consome o gerador de segmentos
            collected_segments = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    collected_segments.append(text)

            transcription_text = " ".join(collected_segments).strip()
            detected_language = info.language or "desconhecido"

            return transcription_text, detected_language

        except (ValueError, FileNotFoundError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Falha durante a transcrição do áudio '{clean_path}': {str(exc)}"
            ) from exc

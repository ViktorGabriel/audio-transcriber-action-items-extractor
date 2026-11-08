"""
Modelos de dados Pydantic (contratos de entrada e saída) para o pipeline
de transcrição e extração semântica de itens de ação.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PriorityLevel(str, Enum):
    """Níveis de prioridade para itens de ação."""
    LOW = "baixa"
    MEDIUM = "media"
    HIGH = "alta"


class ActionItem(BaseModel):
    """Representa um item de ação extraído da reunião."""
    task: str = Field(..., description="Descrição objetiva da tarefa a ser executada")
    assignee: Optional[str] = Field(None, description="Nome do responsável pela execução da tarefa")
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM, description="Prioridade estimada da ação")
    deadline: Optional[str] = Field(None, description="Prazo limite ou data de entrega mencionada")
    context: Optional[str] = Field(None, description="Trecho de contexto ou justificativa mencionada no áudio")


class TranscriptionSegment(BaseModel):
    """Segmento individual de transcrição com carimbo de tempo."""
    id: int = Field(..., description="Índice sequencial do segmento")
    start: float = Field(..., description="Tempo inicial em segundos")
    end: float = Field(..., description="Tempo final em segundos")
    text: str = Field(..., description="Texto transcrito no intervalo")


class TranscriptionResult(BaseModel):
    """Resultado consolidado da transcrição de áudio."""
    raw_text: str = Field(..., description="Texto integral transcrito")
    language: str = Field(..., description="Idioma detectado ou configurado (ex: 'pt', 'en')")
    duration_seconds: float = Field(..., ge=0.0, description="Duração total do áudio em segundos")
    segments: List[TranscriptionSegment] = Field(
        default_factory=list, description="Lista de segmentos com timestamps"
    )


class ExtractionResult(BaseModel):
    """Resultado da análise semântica e extração de ações."""
    summary: str = Field(..., description="Resumo executivo dos pontos abordados no áudio")
    action_items: List[ActionItem] = Field(
        default_factory=list, description="Lista de tarefas acionáveis detectadas"
    )
    key_decisions: List[str] = Field(
        default_factory=list, description="Principais decisões acordadas durante a reunião"
    )


class AnalysisResponse(BaseModel):
    """Contrato final retornado para a interface do usuário."""
    transcription: TranscriptionResult = Field(..., description="Dados da transcrição")
    extraction: ExtractionResult = Field(..., description="Dados da extração semântica")

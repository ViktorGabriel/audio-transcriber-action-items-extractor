"""
Modelos de dados Pydantic (contratos de entrada e saída) para o pipeline
de transcrição e extração semântica de itens de ação.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    """
    Representa um item de ação extraído da transcrição da reunião.
    """
    task: str = Field(..., description="Descrição objetiva da tarefa a ser executada")
    assignee: Optional[str] = Field(None, description="Nome do responsável pela execução")
    priority: Optional[str] = Field(
        default="media",
        description="Prioridade ou nível de urgência da tarefa (ex: baixa, media, alta)"
    )
    deadline_hint: Optional[str] = Field(
        None,
        description="Prazo estimado ou menção temporal de entrega identificada"
    )

    # Propriedade de compatibilidade reversa
    @property
    def deadline(self) -> Optional[str]:
        return self.deadline_hint


class ExtractionResult(BaseModel):
    """
    Resultado estruturado da análise semântica e extração de ações.
    """
    summary: str = Field(..., description="Resumo executivo do conteúdo transcrito")
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Lista de itens de ação identificados"
    )
    language: str = Field(
        default="pt",
        description="Idioma identificado da transcrição e da extração"
    )
    key_decisions: List[str] = Field(
        default_factory=list,
        description="Principais decisões tomadas ou consensos registrados"
    )

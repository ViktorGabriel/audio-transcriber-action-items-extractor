"""
Serviço de análise semântica e extração de itens de ação via LLM.
Suporta Hugging Face Serverless Inference API e fallback local com Qwen/Qwen2.5-1.5B-Instruct.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.schemas import ExtractionResult

load_dotenv()

# Configuração de logging estruturado
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
    )

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = """Você é um especialista em sumarização executiva e extração de itens de ação.
Sua missão é analisar a transcrição de áudio fornecida e retornar ESTREITAMENTE um objeto JSON válido,
sem texto introdutório, sem explicações adicionais e sem blocos markdown fora do JSON.

O JSON deve seguir rigorosamente o seguinte schema:
{
  "summary": "Resumo executivo claro e conciso dos pontos centrais abordados",
  "action_items": [
    {
      "task": "Descrição clara da ação que deve ser executada",
      "assignee": "Nome da pessoa ou departamento responsável (ou null se não especificado)",
      "priority": "baixa, media ou alta",
      "deadline_hint": "Data, prazo ou menção temporal (ou null se não especificado)"
    }
  ],
  "language": "Código do idioma identificado (ex: pt, en)",
  "key_decisions": [
    "Decisões tomadas ou acordos consolidados durante a reunião"
  ]
}
"""

FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": (
            'Transcrição (idioma pt):\n'
            '"Pessoal, na reunião de alinhamento o Carlos ficou de ajustar as rotas da API até sexta-feira. '
            'Eu vou atualizar a documentação do projeto ainda hoje. Ficou acordado que a release para produção '
            'só acontecerá após os testes de carga."'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "summary": (
                    "Alinhamento técnico sobre ajustes na API, atualização da documentação e critérios de deploy."
                ),
                "action_items": [
                    {
                        "task": "Ajustar as rotas da API",
                        "assignee": "Carlos",
                        "priority": "alta",
                        "deadline_hint": "Até sexta-feira",
                    },
                    {
                        "task": "Atualizar a documentação do projeto",
                        "assignee": "Orador / Responsável pela doc",
                        "priority": "media",
                        "deadline_hint": "Ainda hoje",
                    },
                ],
                "language": "pt",
                "key_decisions": [
                    "A release para produção só ocorrerá após a validação completa dos testes de carga."
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    },
]


def sanitize_json_response(raw_text: str) -> str:
    """
    Remove marcações de código markdown (```json ... ```) e caracteres espúrios
    ao redor da estrutura JSON retornada pelo modelo de linguagem.

    Args:
        raw_text: Saída bruta de texto gerada pelo modelo.

    Returns:
        str: String contendo exclusivamente o bloco JSON delimitado por '{' e '}'.
    """
    if not raw_text or not isinstance(raw_text, str):
        return "{}"

    cleaned = raw_text.strip()

    # Remove cercas de código Markdown (```json ... ``` ou ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Localiza o primeiro '{' e o último '}'
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return cleaned[start_idx : end_idx + 1]

    return cleaned


class ActionItemExtractor:
    """
    Serviço de extração de itens de ação e sumários corporativos.

    Utiliza primordialmente a Hugging Face Serverless Inference API com token de autenticação,
    e conta com fallback automático para pipeline local via Transformers.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        hf_token: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
    ) -> None:
        """
        Inicializa o extrator semântico.

        Args:
            model_name: Checkpoint do modelo no Hugging Face Hub.
            hf_token: Token de acesso do Hugging Face (ou lido de HF_TOKEN no env).
            temperature: Temperatura para controle de determinismo da geração.
            max_new_tokens: Limite máximo de novos tokens gerados.
        """
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._inference_client: Optional[InferenceClient] = None
        self._local_pipeline: Optional[Any] = None

    def _get_inference_client(self) -> Optional[InferenceClient]:
        """Inicializa e retorna o cliente da Hugging Face Inference API."""
        if self._inference_client is None and self.hf_token:
            try:
                self._inference_client = InferenceClient(
                    token=self.hf_token,
                    timeout=45.0,
                )
                logger.info(
                    "Cliente Hugging Face Serverless Inference inicializado com sucesso para o modelo: %s",
                    self.model_name,
                )
            except Exception as exc:
                logger.warning(
                    "Não foi possível inicializar o InferenceClient do Hugging Face: %s",
                    exc,
                )
        return self._inference_client

    def _get_local_pipeline(self) -> Any:
        """Inicializa sob demanda (Lazy Loading) o pipeline local de Transformers."""
        if self._local_pipeline is None:
            logger.info(
                "Carregando modelo local via Transformers (CPU): %s...",
                self.model_name,
            )
            load_start = time.perf_counter()
            from transformers import pipeline

            self._local_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device="cpu",
            )
            elapsed = time.perf_counter() - load_start
            logger.info("Modelo local carregado em %.2f segundos.", elapsed)
        return self._local_pipeline

    def _build_messages(self, transcript: str, language: str) -> List[Dict[str, str]]:
        """Monta o histórico de mensagens com System Prompt, Few-Shot e entrada do usuário."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)

        user_content = (
            f"Transcrição (idioma {language}):\n"
            f'"{transcript.strip()}"\n\n'
            "Retorne exclusivamente o JSON estruturado no formato definido:"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    def extract_actions(
        self,
        transcript: str,
        language: str = "pt",
    ) -> ExtractionResult:
        """
        Analisa a transcrição fornecida e extrai o resumo e a lista estruturada de tarefas.

        Args:
            transcript: Texto bruto da transcrição de áudio.
            language: Idioma detectado ou configurado da conversa.

        Returns:
            ExtractionResult: Objeto tipado com resumo, action_items e metadados.

        Raises:
            ValueError: Se o texto fornecido for nulo ou vazio.
        """
        if not transcript or not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("O texto de transcrição não pode ser vazio para extração.")

        clean_transcript = transcript.strip()
        logger.info(
            "Iniciando extração de ações (tamanho: %d caracteres, idioma: %s)...",
            len(clean_transcript),
            language,
        )
        start_time = time.perf_counter()

        messages = self._build_messages(clean_transcript, language)
        raw_output = ""
        client = self._get_inference_client()

        # 1. Tentativa primária: Serverless Inference API do Hugging Face
        if client is not None:
            try:
                logger.info(
                    "Enviando requisição para Hugging Face Inference API (%s)...",
                    self.model_name,
                )
                response = client.chat_completion(
                    messages=messages,
                    model=self.model_name,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                raw_output = response.choices[0].message.content or ""
                logger.info("Resposta recebida da Inference API com sucesso.")
            except Exception as api_err:
                logger.warning(
                    "Falha na Hugging Face Inference API: %s. Acionando fallback local.",
                    api_err,
                )
                raw_output = ""

        # 2. Fallback local via Transformers
        if not raw_output:
            try:
                local_pipe = self._get_local_pipeline()
                logger.info("Executando inferência localmente via Transformers...")
                prompt_text = f"{SYSTEM_PROMPT}\n\nTranscrição:\n{clean_transcript}\n\nJSON:"
                results = local_pipe(
                    prompt_text,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=False,
                )
                raw_output = results[0]["generated_text"][len(prompt_text) :].strip()
            except Exception as local_err:
                logger.error("Falha na inferência local com Transformers: %s", local_err)

        # 3. Sanitização e Parsing Pydantic
        clean_json_str = sanitize_json_response(raw_output)
        inference_time = time.perf_counter() - start_time

        try:
            parsed_dict = json.loads(clean_json_str)
            if not isinstance(parsed_dict, dict):
                raise ValueError("JSON parseado não é um objeto chave-valor.")

            if "language" not in parsed_dict or not parsed_dict["language"]:
                parsed_dict["language"] = language

            result = ExtractionResult.model_validate(parsed_dict)
            logger.info(
                "Extração concluída com sucesso em %.2fs. Ações extraídas: %d.",
                inference_time,
                len(result.action_items),
            )
            return result

        except Exception as parse_exc:
            logger.warning(
                "Falha ao validar JSON estrito (%s). Construindo fallback defensivo. Saída bruta: %s",
                parse_exc,
                raw_output[:200],
            )
            fallback_summary = (
                clean_transcript[:250] + "..."
                if len(clean_transcript) > 250
                else clean_transcript
            )
            return ExtractionResult(
                summary=f"Sumário preliminar: {fallback_summary}",
                action_items=[],
                language=language,
                key_decisions=[],
            )

    def extract(
        self,
        transcription_text: str,
        custom_instructions: Optional[str] = None,
    ) -> ExtractionResult:
        """Método de compatibilidade para a interface de extração."""
        return self.extract_actions(transcript=transcription_text, language="pt")


def extract_actions(transcript: str, language: str = "pt") -> ExtractionResult:
    """
    Função utilitária direta para extração de itens de ação e resumo.

    Args:
        transcript: Texto transcrito da conversa.
        language: Código do idioma detectado (default: 'pt').

    Returns:
        ExtractionResult: Dados estruturados validados por Pydantic.
    """
    extractor = ActionItemExtractor()
    return extractor.extract_actions(transcript=transcript, language=language)

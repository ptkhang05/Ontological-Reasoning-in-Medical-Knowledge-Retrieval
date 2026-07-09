from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from clinical_nlp.schemas import ExternalEntity

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You extract Vietnamese clinical concepts for Viettel Medical 2026.
Return only a JSON array. Each item must have:
- text: exact substring copied from the input
- position: [start, end] zero-based character offsets, end exclusive
- type: one of TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC

Use short exact clinical spans only. Never output section headings such as
"Tiền sử bệnh", "Bệnh sử hiện tại", "Triệu chứng hiện tại", or "Đánh giá".
Do not include patient identifiers. Do not add explanations. Do not invent spans."""


@dataclass(frozen=True)
class LocalLlmConfig:
    base_url: str
    model: str = "qwen2.5-7b-instruct"
    api_key: str | None = None
    timeout_seconds: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 4096


class LocalOpenAICompatibleExtractor:
    """Optional extractor for self-hosted OpenAI-compatible LLM servers.

    This adapter is intentionally conservative: it only returns raw external
    entity proposals. The main pipeline remains responsible for offset
    validation, type validation, terminology lookup, and overlap filtering.
    """

    def __init__(
        self,
        config: LocalLlmConfig,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt

    def extract(self, text: str) -> list[ExternalEntity]:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            logger.warning("Local LLM extractor failed: %s", type(exc).__name__)
            return []

        content = _message_content(response_payload)
        if content is None:
            return []
        return parse_llm_entities(content)

    def _chat_completions_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers


def build_external_extractor_from_env() -> LocalOpenAICompatibleExtractor | None:
    base_url = os.getenv("CLINICAL_NLP_LOCAL_LLM_BASE_URL", "").strip()
    if not base_url:
        return None

    return LocalOpenAICompatibleExtractor(
        LocalLlmConfig(
            base_url=base_url,
            model=os.getenv("CLINICAL_NLP_LOCAL_LLM_MODEL", "qwen2.5-7b-instruct"),
            api_key=os.getenv("CLINICAL_NLP_LOCAL_LLM_API_KEY") or None,
            timeout_seconds=_float_env("CLINICAL_NLP_LOCAL_LLM_TIMEOUT", 120.0),
            temperature=_float_env("CLINICAL_NLP_LOCAL_LLM_TEMPERATURE", 0.0),
            max_tokens=_int_env("CLINICAL_NLP_LOCAL_LLM_MAX_TOKENS", 4096),
        )
    )


def parse_llm_entities(content: str) -> list[ExternalEntity]:
    parsed = _parse_json_payload(content)
    if isinstance(parsed, dict):
        parsed = parsed.get("entities")
    if not isinstance(parsed, list):
        return []

    entities: list[ExternalEntity] = []
    for item in parsed:
        if isinstance(item, dict):
            entities.append(dict(item))
    return entities


def _parse_json_payload(content: str) -> Any:
    stripped = content.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.S | re.I)
    if fenced is not None:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[[\s\S]*\]", stripped)
    if array_match is None:
        return None
    try:
        return json.loads(array_match.group(0))
    except json.JSONDecodeError:
        return None


def _message_content(response_payload: Any) -> str | None:
    if not isinstance(response_payload, dict):
        return None
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    content = first.get("text")
    if isinstance(content, str):
        return content
    return None


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

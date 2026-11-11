"""Minimal LARM client used to summarize meeting transcriptions.

LARM exposes an OpenAI compatible chat completions API, so this module reuses the
``openai`` client already declared in the project requirements instead of adding a new
HTTP dependency.  LARM owns the model lifecycle (loading, leases, unload): py-record
only sends one request and reads the answer back.
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional

LARM_DEFAULT_BASE_URL = "http://127.0.0.1:8090/v1"
LARM_DEFAULT_TIMEOUT = 600.0
LARM_DEFAULT_API_KEY = "local"
# Bounded retry: the request may fail once while the model is still warming up.
LARM_MAX_RETRIES = 1

LARM_SUMMARY_SYSTEM_PROMPT = (
    "Eres un asistente especializado en resumir transcripciones de reuniones.\n"
    "\n"
    "Instrucciones:\n"
    "1. Genera un resumen fiel al contenido proporcionado.\n"
    "2. Identifica los temas principales tratados en la reunión.\n"
    "3. Describe los puntos relevantes que se discutieron.\n"
    "4. No inventes hechos, participantes, decisiones ni conclusiones.\n"
    "5. No incluyas información que no aparezca en la transcripción.\n"
    "6. Mantén los nombres técnicos, identificadores, rutas, comandos y términos "
    "relevantes tal y como aparecen en la transcripción.\n"
    "7. Si la transcripción es insuficiente, ambigua o incompleta, indícalo de forma "
    "explícita en lugar de suponer lo que ocurrió.\n"
    "\n"
    "Formato de salida:\n"
    "- Devuelve texto plano o Markdown sencillo.\n"
    "- No realices extracción estructurada de tareas, responsables ni decisiones.\n"
)

LARM_LANGUAGE_DIRECTIVES = {
    "es": "Redacta el resumen en español.",
    "en": "Redacta el resumen en inglés.",
}


class LarmError(RuntimeError):
    """Raised when LARM cannot deliver a summary.

    It inherits from ``RuntimeError`` so the CLI reports it as a controlled failure
    (exit code 1) instead of an unexpected crash.
    """


@dataclass(frozen=True)
class LarmConfig:
    """Connection settings for a LARM instance."""

    base_url: str
    model: str
    timeout: float
    api_key: str
    max_retries: int = LARM_MAX_RETRIES


def build_system_prompt(lang: str = "en") -> str:
    """Return the meeting summary system prompt for the requested language."""
    directive = LARM_LANGUAGE_DIRECTIVES.get((lang or "en").lower())
    if directive is None:
        directive = f"Redacta el resumen en el idioma «{lang}»."
    return f"{LARM_SUMMARY_SYSTEM_PROMPT}{directive}\n"


def normalize_base_url(raw_url: str) -> str:
    """Normalize the base URL and default the API path to ``/v1`` when missing."""
    base_url = (raw_url or "").strip().rstrip("/")
    if not base_url:
        raise LarmError(f"Invalid LARM base URL: '{raw_url}'.")
    if not re.match(r"^https?://", base_url, re.IGNORECASE):
        raise LarmError(
            f"Invalid LARM base URL '{raw_url}': it must start with http:// or https://."
        )
    if "/" not in base_url.split("://", 1)[1]:
        base_url = f"{base_url}/v1"
    return base_url


def resolve_config(model: Optional[str] = None, env=None) -> LarmConfig:
    """Build the LARM configuration from the CLI argument and the environment.

    The model alias is mandatory: LARM registers the aliases per installation, so
    py-record never assumes a fixed one.
    """
    environment = os.environ if env is None else env

    base_url = normalize_base_url(environment.get("LARM_BASE_URL") or LARM_DEFAULT_BASE_URL)

    alias = (model or "").strip() or (environment.get("LARM_MODEL") or "").strip()
    if not alias:
        raise LarmError(
            "No LARM model configured. Set the LARM_MODEL environment variable or pass "
            "--larm-model with an alias registered in LARM."
        )

    raw_timeout = (environment.get("LARM_TIMEOUT") or "").strip()
    if raw_timeout:
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise LarmError(
                f"Invalid LARM_TIMEOUT '{raw_timeout}': it must be a number of seconds."
            ) from exc
        if timeout <= 0:
            raise LarmError(
                f"Invalid LARM_TIMEOUT '{raw_timeout}': it must be greater than zero."
            )
    else:
        timeout = LARM_DEFAULT_TIMEOUT

    api_key = (environment.get("LARM_API_KEY") or "").strip() or LARM_DEFAULT_API_KEY

    return LarmConfig(
        base_url=base_url,
        model=alias,
        timeout=timeout,
        api_key=api_key,
    )


def build_client(config: LarmConfig):
    """Create the OpenAI compatible client pointed at the LARM instance."""
    from openai import OpenAI

    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout,
        max_retries=config.max_retries,
    )


def _get(obj, key, default=None):
    """Read a field from an object or from a plain mapping."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def clean_summary_text(raw) -> str:
    """Normalize the model answer into clean plain text."""
    if raw is None:
        return ""
    text = str(raw).replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def _error_message(exc) -> str:
    """Extract the most descriptive message available in an API error."""
    message = _get(exc, "message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    response = _get(exc, "response")
    body = _get(response, "text") if response is not None else None
    if isinstance(body, str) and body.strip():
        return body.strip()
    return str(exc)


def _looks_like_context_overflow(message: str) -> bool:
    """Detect server messages caused by a transcript larger than the context window."""
    haystack = (message or "").lower()
    markers = (
        "context",
        "context_length",
        "context window",
        "too long",
        "maximum length",
        "max_tokens",
        "exceed",
        "n_ctx",
        "kv cache",
    )
    return any(marker in haystack for marker in markers)


def _translate_error(exc, config: LarmConfig) -> LarmError:
    """Map an API failure into a comprehensible LarmError."""
    timeout_types = ()
    connection_types = ()
    try:
        from openai import APIConnectionError, APITimeoutError

        timeout_types = (APITimeoutError,)
        connection_types = (APIConnectionError,)
    except Exception:
        timeout_types = ()
        connection_types = ()

    if (isinstance(exc, timeout_types) or isinstance(exc, TimeoutError)
            or "Timeout" in type(exc).__name__):
        return LarmError(
            f"LARM timed out after {config.timeout:g}s while summarizing with model "
            f"'{config.model}'. The model may still be loading, or the transcript may "
            "exceed its context window."
        )

    status_code = _get(exc, "status_code")
    if status_code is not None:
        message = _error_message(exc)
        if status_code == 404:
            return LarmError(
                f"LARM model '{config.model}' was not found at {config.base_url}. "
                "Check the aliases registered in that LARM instance (GET /v1/models)."
            )
        if _looks_like_context_overflow(message):
            return LarmError(
                f"The transcription exceeds the context window of LARM model "
                f"'{config.model}': {message}. Processing the transcript in segments is "
                "not implemented yet; split the transcript manually and summarize it in "
                "parts."
            )
        return LarmError(
            f"LARM returned HTTP {status_code} while summarizing with model "
            f"'{config.model}': {message}"
        )

    if isinstance(exc, connection_types) or "Connection" in type(exc).__name__:
        return LarmError(
            f"Could not connect to LARM at {config.base_url}. Check that the instance is "
            f"running and reachable ({exc})."
        )

    return LarmError(
        f"Unexpected error while requesting the summary from LARM at {config.base_url}: {exc}"
    )


def summarize_transcription(
    transcription: str,
    config: LarmConfig,
    system_prompt: Optional[str] = None,
) -> str:
    """Send the transcription to LARM and return the cleaned summary.

    The whole transcription is sent as the user message: it is never truncated
    silently. Optional sampling parameters are not used because not every local
    model behind LARM accepts them.
    """
    if not transcription or not transcription.strip():
        raise LarmError("The transcription is empty; there is nothing to summarize.")

    client = build_client(config)
    messages = [
        {"role": "system", "content": system_prompt or build_system_prompt()},
        {"role": "user", "content": transcription},
    ]

    try:
        completion = client.chat.completions.create(
            model=config.model,
            messages=messages,
        )
    except Exception as exc:
        raise _translate_error(exc, config) from exc

    choices = _get(completion, "choices")
    if not choices:
        raise LarmError(
            f"LARM returned no choices for model '{config.model}'; the response was empty."
        )

    message = _get(choices[0], "message")
    content = _get(message, "content")
    summary = clean_summary_text(content)
    if not summary:
        raise LarmError(
            f"LARM returned an empty summary for model '{config.model}'; the response "
            "contained no text."
        )
    return summary


def list_models(config: LarmConfig) -> List[str]:
    """Return the model aliases registered in the LARM instance."""
    client = build_client(config)
    try:
        page = client.models.list()
    except Exception as exc:
        raise _translate_error(exc, config) from exc

    aliases: List[str] = []
    for entry in _get(page, "data") or []:
        model_id = _get(entry, "id")
        if model_id:
            aliases.append(model_id)
    return aliases

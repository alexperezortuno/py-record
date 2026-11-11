"""Structured analysis of meeting transcriptions with a local model served by LARM.

This module is independent from the recording, transcription and summary flows: it
reads an existing transcription, asks LARM for a structured extraction (topics,
decisions, action items and open questions), validates that answer against the
transcription itself and exports the result as JSON and/or Markdown.

The answer of the model is never accepted as-is:

* the payload must be a JSON object that follows the documented schema;
* every element must carry a unique identifier and at least one evidence entry;
* every quote must exist literally in the transcription (only whitespace is
  normalized) and, when the transcript carries time marks, the declared
  timestamps must match the source segment that contains the quote.

An element whose evidence cannot be verified is rejected and the problem is
reported: no alternative quote is ever fabricated.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from meeting import larm

# Version of the structure exported by this module.
ANALYSIS_SCHEMA_VERSION = "1.0"

# Tolerance, in seconds, when a timestamp declared by the model is compared with
# the time mark of the source segment that contains the quote.
TIMESTAMP_TOLERANCE_SECONDS = 1.0

# Collections of the analysis, in the order used by the prompt and the exports.
COLLECTION_TOPICS = "topics"
COLLECTION_DECISIONS = "decisions"
COLLECTION_ACTION_ITEMS = "action_items"
COLLECTION_OPEN_QUESTIONS = "open_questions"

COLLECTIONS = (
    COLLECTION_TOPICS,
    COLLECTION_DECISIONS,
    COLLECTION_ACTION_ITEMS,
    COLLECTION_OPEN_QUESTIONS,
)

OUTPUT_FORMATS = ("json", "markdown", "both")

ANALYSIS_SYSTEM_PROMPT = (
    "Eres un asistente especializado en extraer información estructurada a partir "
    "de transcripciones de reuniones.\n"
    "\n"
    "Reglas obligatorias:\n"
    "1. La transcripción que recibes es la ÚNICA fuente de información autorizada. "
    "No uses conocimiento externo, no completes datos y no inventes nombres, fechas, "
    "decisiones ni tareas.\n"
    "2. Conserva el idioma original de la transcripción en todas las descripciones.\n"
    "3. Distingue con cuidado entre las categorías:\n"
    "   - topics: los hechos y asuntos tratados en la reunión.\n"
    "   - decisions: acuerdos confirmados de forma explícita. No conviertas "
    "sugerencias, hipótesis ni preguntas en decisiones.\n"
    "   - action_items: tareas que alguien asume explícitamente.\n"
    "   - open_questions: preguntas planteadas que no reciben respuesta.\n"
    "4. assignee es null salvo que la transcripción nombre explícitamente a la "
    "persona responsable. No lo deduzcas por contexto.\n"
    "5. due_date es null salvo que la transcripción contenga un plazo explícito. No "
    "lo deduzcas.\n"
    "6. Cada elemento incluye evidencia con una cita literal (quote) copiada "
    "exactamente de la transcripción, sin parafrasear, resumir ni corregir. Si la "
    "cita está en una línea con marca de tiempo, copia el valor exacto de esa "
    "marca, convertido a segundos (multiplica los minutos por 60 y suma los "
    "segundos: 00:00 = 0, 04:48 = 4*60+48 = 288, 06:11 = 6*60+11 = 371):\n"
    "   - línea con la marca [04:48] -> start_seconds 288.0 y end_seconds 288.0\n"
    "   - línea con la marca [1.25 --> 2.50] -> start_seconds 1.25 y "
    "end_seconds 2.50\n"
    "   No inventes duraciones: si la marca tiene un solo tiempo, repite ese mismo "
    "valor en start_seconds y en end_seconds. Si no estás seguro del valor exacto "
    "de la marca, deja start_seconds y end_seconds en null: es preferible dejar la "
    "marca en null que entregar un valor equivocado. Si la transcripción no tiene "
    "marcas de tiempo, deja ambos en null.\n"
    "7. Si una categoría no tiene elementos, devuelve un arreglo vacío.\n"
    "8. Si la transcripción es insuficiente para una categoría, devuelve esa "
    "categoría vacía en lugar de suponer.\n"
    "\n"
    "Formato de salida:\n"
    "- Devuelve únicamente un objeto JSON válido, sin texto adicional y sin bloques "
    "de código.\n"
    "- Respeta exactamente esta estructura y estos nombres de campo:\n"
    "{\n"
    '  "schema_version": "1.0",\n'
    '  "topics": [{"id": "T1", "title": "titulo", "description": "descripcion", '
    '"evidence": [{"quote": "cita literal", "start_seconds": null, '
    '"end_seconds": null}]}],\n'
    '  "decisions": [{"id": "D1", "description": "descripcion", '
    '"evidence": [{"quote": "cita literal", "start_seconds": null, '
    '"end_seconds": null}]}],\n'
    '  "action_items": [{"id": "A1", "description": "descripcion", "assignee": null, '
    '"due_date": null, "evidence": [{"quote": "cita literal", '
    '"start_seconds": null, "end_seconds": null}]}],\n'
    '  "open_questions": [{"id": "Q1", "question": "pregunta", '
    '"evidence": [{"quote": "cita literal", "start_seconds": null, '
    '"end_seconds": null}]}]\n'
    "}\n"
)


class AnalysisError(RuntimeError):
    """Raised when the analysis cannot be produced.

    It inherits from ``RuntimeError`` so the CLI reports it as a controlled
    failure (exit code 1) instead of an unexpected crash.
    """


# =====================================================================================
# Data model
# =====================================================================================
@dataclass(frozen=True)
class Evidence:
    """A literal fragment of the transcription backing an extracted element."""

    quote: str
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quote": self.quote,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
        }


@dataclass(frozen=True)
class Topic:
    """A matter discussed during the meeting."""

    id: str
    title: str
    description: str
    evidence: Tuple[Evidence, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class Decision:
    """An agreement explicitly confirmed during the meeting."""

    id: str
    description: str
    evidence: Tuple[Evidence, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class ActionItem:
    """A task explicitly assumed during the meeting."""

    id: str
    description: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None
    evidence: Tuple[Evidence, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "assignee": self.assignee,
            "due_date": self.due_date,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class OpenQuestion:
    """A question raised during the meeting that was not answered."""

    id: str
    question: str
    evidence: Tuple[Evidence, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class MeetingAnalysis:
    """The validated result of the analysis of one transcription."""

    topics: Tuple[Topic, ...] = ()
    decisions: Tuple[Decision, ...] = ()
    action_items: Tuple[ActionItem, ...] = ()
    open_questions: Tuple[OpenQuestion, ...] = ()
    schema_version: str = ANALYSIS_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topics": [item.to_dict() for item in self.topics],
            "decisions": [item.to_dict() for item in self.decisions],
            "action_items": [item.to_dict() for item in self.action_items],
            "open_questions": [item.to_dict() for item in self.open_questions],
        }

    def count(self) -> int:
        """Return the number of extracted elements."""
        return (len(self.topics) + len(self.decisions) + len(self.action_items)
                + len(self.open_questions))


@dataclass(frozen=True)
class ValidationProblem:
    """A problem found while validating the answer of the model."""

    collection: str
    item_id: str
    message: str

    def __str__(self) -> str:
        return f"[{self.collection}] {self.item_id}: {self.message}"


@dataclass
class AnalysisResult:
    """The validated analysis together with the problems found on the way."""

    analysis: MeetingAnalysis
    problems: List[ValidationProblem] = field(default_factory=list)

    @property
    def rejected(self) -> int:
        """Return how many elements were rejected during the validation."""
        return len(self.problems)


# =====================================================================================
# Transcript reading and indexing
# =====================================================================================
@dataclass(frozen=True)
class SourceSegment:
    """A fragment of the transcript that carries a time mark."""

    start: float
    end: Optional[float]
    norm_start: int
    norm_end: int


# Time mark written as a range: "[1.25 --> 2.50] text" (Faster-Whisper / WhisperX).
_SEGMENT_RE = re.compile(
    r"^\s*\[(?P<start>[^\]]+?)\s*-->\s*(?P<end>[^\]]+?)\]\s*(?P<body>.*)$"
)

# Single time mark: "[00:00]" or "[04:48]", usually after the speaker label.
_SINGLE_MARK_RE = re.compile(
    r"\[(?P<value>\d{1,3}(?::\d{1,2}){0,2}(?:\.\d+)?)\]"
)


def read_transcript(path: str) -> str:
    """Read a transcription without modifying it.

    Parameters:
        path: Path of the transcription file.

    Returns:
        str: The full content of the transcription.

    Raises:
        FileNotFoundError: If the transcription does not exist.
        AnalysisError: If the file is empty, is not UTF-8 text or cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    if os.path.isdir(path):
        raise AnalysisError(f"The transcript path is a directory: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError as exc:
        raise AnalysisError(
            f"The transcript '{path}' is not valid UTF-8 text ({exc}). Convert the file "
            "to UTF-8 before analyzing it; the original file is left untouched."
        ) from exc
    except OSError as exc:
        raise AnalysisError(f"Could not read the transcript '{path}': {exc}") from exc

    if not text.strip():
        raise AnalysisError(
            f"The transcript '{path}' is empty; there is nothing to analyze."
        )
    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs into single spaces (minimal normalization)."""
    return " ".join((text or "").split())


def _parse_timestamp_value(raw: str) -> Optional[float]:
    """Parse a time mark written as seconds or as ``hh:mm:ss``/``mm:ss``."""
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    parts = value.split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60.0 + number
    return seconds


def _index_transcript(transcript: str) -> Tuple[str, List[SourceSegment]]:
    """Normalize the transcript and locate every timestamped segment.

    Returns:
        tuple: The normalized transcript and the list of its timestamped segments.
    """
    lines: List[str] = []
    segments: List[SourceSegment] = []
    offset = 0

    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _SEGMENT_RE.match(line)
        single = None
        if match is not None:
            body = match.group("body")
            prefix = line[: match.start("body")]
            start_seconds = _parse_timestamp_value(match.group("start"))
            end_seconds = _parse_timestamp_value(match.group("end"))
            if start_seconds is None or end_seconds is None:
                start_seconds = None
                end_seconds = None
        else:
            body = line
            prefix = ""
            start_seconds = None
            end_seconds = None
            single = _SINGLE_MARK_RE.search(line)
            if single is not None:
                start_seconds = _parse_timestamp_value(single.group("value"))

        prefix_norm = _normalize_whitespace(prefix)
        body_norm = _normalize_whitespace(body)

        if prefix_norm and body_norm:
            line_norm = f"{prefix_norm} {body_norm}"
            body_start = offset + len(prefix_norm) + 1
            body_end = body_start + len(body_norm)
        elif prefix_norm:
            line_norm = prefix_norm
            body_start = offset
            body_end = offset
        else:
            line_norm = body_norm
            body_start = offset
            body_end = offset + len(body_norm)

        if start_seconds is not None and body_end > body_start:
            # A single time mark does not close the segment: its end is the start of
            # the next marked segment, and it stays unknown when there is none.
            segments.append(SourceSegment(start_seconds, end_seconds, body_start,
                                          body_end))

        lines.append(line_norm)
        offset += len(line_norm) + 1  # the join separator

    normalized = " ".join(lines)
    segments = _close_open_segments(segments)
    return normalized, segments


def _close_open_segments(segments: List[SourceSegment]) -> List[SourceSegment]:
    """Complete the segments that only carry a start mark with the next start."""
    closed: List[SourceSegment] = []
    for position, segment in enumerate(segments):
        if segment.end is not None:
            closed.append(segment)
            continue
        end = None
        for following in segments[position + 1:]:
            if following.start >= segment.start:
                end = following.start
                break
        closed.append(SourceSegment(segment.start, end, segment.norm_start,
                                    segment.norm_end))
    return closed


def _locate_quotes(normalized: str, quote: str) -> List[Tuple[int, int]]:
    """Return the offsets of every occurrence of the quote in the transcript."""
    occurrences: List[Tuple[int, int]] = []
    if not quote:
        return occurrences
    position = normalized.find(quote)
    while position >= 0:
        occurrences.append((position, position + len(quote)))
        position = normalized.find(quote, position + 1)
    return occurrences


def _lower_keeping_length(text: str) -> str:
    """Return the text in lower case without changing its length."""
    return "".join(
        char.lower() if len(char.lower()) == 1 else char for char in text
    )


def _segments_for_range(segments: Sequence[SourceSegment],
                         start: int, end: int) -> Tuple[Optional[SourceSegment],
                                                       Optional[SourceSegment]]:
    """Return the first and the last source segment overlapping a text range."""
    overlapping = [
        segment for segment in segments
        if segment.norm_start < end and segment.norm_end > start
    ]
    if not overlapping:
        return None, None
    overlapping.sort(key=lambda segment: segment.norm_start)
    return overlapping[0], overlapping[-1]


# =====================================================================================
# LARM request
# =====================================================================================
def build_analysis_system_prompt() -> str:
    """Return the system prompt used for the structured extraction."""
    return ANALYSIS_SYSTEM_PROMPT


def request_analysis(transcription: str, config: larm.LarmConfig) -> str:
    """Ask LARM for the structured analysis of a transcription.

    The whole transcription is sent as the user message and it is never truncated
    silently. The request reuses the LARM client and its configuration, so the
    model lifecycle (loading, leases, unload) stays under LARM control.

    Parameters:
        transcription: Full content of the transcription.
        config: LARM connection settings.

    Returns:
        str: The raw text returned by the model.

    Raises:
        AnalysisError: If LARM is unreachable, misconfigured or answers with an
            empty response.
    """
    try:
        return larm.summarize_transcription(
            transcription, config, system_prompt=build_analysis_system_prompt()
        )
    except larm.LarmError as exc:
        raise AnalysisError(f"The analysis request to LARM failed: {exc}") from exc


def _json_candidates(text: str) -> List[str]:
    """Return the texts that may hold the JSON payload, best guess first."""
    candidates = [text]
    extracted = _extract_json_object(text)
    if extracted is not None and extracted != text:
        candidates.append(extracted)
    return candidates


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object of the text, if there is one."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def parse_analysis_payload(raw: str) -> Dict[str, Any]:
    """Parse the answer of the model into a JSON object.

    Parameters:
        raw: Text returned by the model.

    Returns:
        dict: The parsed payload.

    Raises:
        AnalysisError: If the answer is empty, is not valid JSON or does not hold
            a JSON object.
    """
    text = (raw or "").strip()
    if not text:
        raise AnalysisError(
            "LARM returned an empty answer; there is nothing to validate."
        )

    errors: List[str] = []
    for candidate in _json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(payload, dict):
            raise AnalysisError(
                "The answer of the model is not a JSON object; the analysis cannot be "
                "validated."
            )
        return payload

    detail = errors[0] if errors else "no JSON object was found"
    raise AnalysisError(f"The answer of the model is not valid JSON: {detail}")


# =====================================================================================
# Validation
# =====================================================================================
def _require_text(value: Any) -> Optional[str]:
    """Return the stripped text of a required string field."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_text(value: Any) -> Optional[str]:
    """Return the stripped text of an optional field, or None when it is null."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("the value must be a string or null")
    text = value.strip()
    return text or None


def _optional_timestamp(value: Any) -> Optional[float]:
    """Return a timestamp of the evidence, or None when it is not present.

    A number is taken as seconds. A string is accepted when it is the time mark
    of the transcription (``mm:ss``) or a plain number of seconds, because the
    model may copy the mark instead of converting it.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("the value must be a number of seconds or null")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if re.match(r"^\d{1,3}:\d{1,2}(?:\.\d+)?$", text):
            number = _parse_timestamp_value(text)
        else:
            try:
                number = float(text)
            except ValueError:
                raise ValueError(
                    "the value must be a number of seconds, a 'mm:ss' time mark or null"
                ) from None
    else:
        raise ValueError("the value must be a number of seconds or null")
    if number is None:
        raise ValueError("the value must be a number of seconds or null")
    if number < 0:
        raise ValueError("the value must not be negative")
    return number


def _timestamp_candidates(value: float) -> List[float]:
    """Return the accepted readings of a declared timestamp.

    The model is asked for a number of seconds, but it may copy the time mark of
    the transcript (``04:48``) and write it as ``4.48``. Both readings are
    offered to the correspondence check, and only the one that matches the source
    segment of the transcript is accepted, so a wrong value is never exported.
    """
    candidates = [float(value)]
    if not float(value).is_integer():
        minutes = int(value)
        seconds = round((value - minutes) * 100)
        if 0 <= minutes <= 59 and 0 <= seconds <= 59:
            candidates.append(minutes * 60.0 + seconds)
    return candidates


def _validate_evidence(entry: Any, normalized: str,
                       segments: Sequence[SourceSegment],
                       has_timestamps: bool) -> Evidence:
    """Validate one evidence entry against the transcription.

    Raises:
        ValueError: If the entry is malformed or cannot be verified.
    """
    if not isinstance(entry, dict):
        raise ValueError("the evidence entry must be an object")

    quote = entry.get("quote")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("the evidence entry requires a non-empty 'quote'")

    try:
        start_seconds = _optional_timestamp(entry.get("start_seconds"))
        end_seconds = _optional_timestamp(entry.get("end_seconds"))
    except ValueError as exc:
        raise ValueError(f"the evidence timestamps are invalid: {exc}") from exc

    if (start_seconds is None) != (end_seconds is None):
        raise ValueError(
            "the evidence timestamps must declare both 'start_seconds' and "
            "'end_seconds', or none of them"
        )
    if start_seconds is not None and end_seconds < start_seconds:
        raise ValueError("'end_seconds' must not be smaller than 'start_seconds'")

    quote_norm = _normalize_whitespace(quote)
    occurrences = _locate_quotes(normalized, quote_norm)
    if not occurrences:
        # The model sometimes lowercases the first letter of the fragment. The
        # words and their order are still the ones of the transcript, so the
        # fragment is accepted and the text of the transcript is the one that is
        # stored: a paraphrase is never accepted as evidence.
        occurrences = _locate_quotes(_lower_keeping_length(normalized),
                                     _lower_keeping_length(quote_norm))
    if not occurrences:
        raise ValueError(
            "the quote does not exist literally in the transcription; a paraphrase "
            "is not accepted as evidence"
        )

    # The exported quote is always the fragment of the transcript, never a text
    # written by the model.
    stored_quote = normalized[occurrences[0][0]:occurrences[0][1]]

    if start_seconds is not None:
        if not has_timestamps:
            raise ValueError(
                "the transcript does not provide time marks, so the declared "
                "timestamps cannot be verified"
            )
        problem = None
        accepted = None
        for position, end_position in occurrences:
            first, last = _segments_for_range(segments, position, end_position)
            if first is None or last is None:
                problem = (
                    "the quote does not belong to a timestamped segment of the "
                    "transcription"
                )
                continue
            for start_candidate in _timestamp_candidates(start_seconds):
                for end_candidate in _timestamp_candidates(end_seconds):
                    issue = _timestamps_problem(first, last, start_candidate,
                                                end_candidate)
                    if issue is None:
                        accepted = (start_candidate, end_candidate)
                        break
                if accepted is not None:
                    break
            if accepted is not None:
                break
            problem = _timestamps_problem(first, last, start_seconds, end_seconds)
        if accepted is None:
            raise ValueError(problem)
        start_seconds, end_seconds = accepted
    else:
        # The model did not declare the timestamps: they are anchored in the
        # segment of the transcription that holds the verified quote. They are
        # the times of the transcript itself, never invented ones.
        start_seconds, end_seconds = _source_times(segments, occurrences)

    return Evidence(quote=stored_quote, start_seconds=start_seconds,
                    end_seconds=end_seconds)


def _source_times(segments: Sequence[SourceSegment],
                  occurrences: Sequence[Tuple[int, int]]) -> Tuple[Optional[float],
                                                                  Optional[float]]:
    """Return the times of the transcript segment that holds the quote."""
    for position, end_position in occurrences:
        first, last = _segments_for_range(segments, position, end_position)
        if first is None or last is None:
            continue
        end = last.end if last.end is not None else last.start
        return first.start, end
    return None, None


def _timestamps_problem(first: SourceSegment, last: SourceSegment,
                         start_seconds: float, end_seconds: float) -> Optional[str]:
    """Return why the declared timestamps do not match the source segments.

    Returns:
        str: The description of the mismatch, or None when they match.
    """
    tolerance = TIMESTAMP_TOLERANCE_SECONDS
    if start_seconds < first.start - tolerance:
        return (f"'start_seconds' ({start_seconds:g}) is before the source segment "
                f"that starts at {first.start:g}")
    if first.end is not None and start_seconds > first.end + tolerance:
        return (f"'start_seconds' ({start_seconds:g}) does not match the source "
                f"segment [{first.start:g}, {first.end:g}]")
    if end_seconds < last.start - tolerance:
        return (f"'end_seconds' ({end_seconds:g}) is before the source segment that "
                f"starts at {last.start:g}")
    if last.end is not None and end_seconds > last.end + tolerance:
        return (f"'end_seconds' ({end_seconds:g}) does not match the source "
                f"segment [{last.start:g}, {last.end:g}]")
    return None


def _validate_elements(collection: str, entries: Any, normalized: str,
                       segments: Sequence[SourceSegment], has_timestamps: bool,
                       problems: List[ValidationProblem]) -> List[Any]:
    """Validate every element of a collection and drop the invalid ones."""
    if not isinstance(entries, list):
        raise AnalysisError(
            f"The collection '{collection}' must be a list; the answer of the model "
            "does not follow the expected schema."
        )

    accepted: List[Any] = []
    seen_ids: Dict[str, int] = {}

    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(ValidationProblem(
                collection, f"#{position + 1}",
                "the element must be an object"))
            continue

        item_id = _require_text(entry.get("id"))
        if item_id is None:
            problems.append(ValidationProblem(
                collection, f"#{position + 1}",
                "the element requires a non-empty 'id'"))
            continue
        if item_id in seen_ids:
            problems.append(ValidationProblem(
                collection, item_id,
                "the identifier is duplicated inside the collection"))
            continue
        seen_ids[item_id] = position

        try:
            element = _build_element(collection, item_id, entry, normalized, segments,
                                     has_timestamps)
        except ValueError as exc:
            problems.append(ValidationProblem(collection, item_id, str(exc)))
            continue
        accepted.append(element)

    return accepted


def _build_element(collection: str, item_id: str, entry: Dict[str, Any],
                   normalized: str, segments: Sequence[SourceSegment],
                   has_timestamps: bool) -> Any:
    """Build one validated element of the analysis."""
    evidence_entries = entry.get("evidence")
    if not isinstance(evidence_entries, list):
        raise ValueError("the element requires an 'evidence' list")
    if not evidence_entries:
        raise ValueError("the element requires at least one evidence entry")

    evidence: List[Evidence] = []
    for evidence_entry in evidence_entries:
        evidence.append(_validate_evidence(evidence_entry, normalized, segments,
                                           has_timestamps))

    if collection == COLLECTION_TOPICS:
        title = _require_text(entry.get("title"))
        description = _require_text(entry.get("description"))
        if title is None:
            raise ValueError("the element requires a non-empty 'title'")
        if description is None:
            raise ValueError("the element requires a non-empty 'description'")
        return Topic(id=item_id, title=title, description=description,
                     evidence=tuple(evidence))

    if collection == COLLECTION_DECISIONS:
        description = _require_text(entry.get("description"))
        if description is None:
            raise ValueError("the element requires a non-empty 'description'")
        return Decision(id=item_id, description=description,
                        evidence=tuple(evidence))

    if collection == COLLECTION_ACTION_ITEMS:
        description = _require_text(entry.get("description"))
        if description is None:
            raise ValueError("the element requires a non-empty 'description'")
        try:
            assignee = _optional_text(entry.get("assignee"))
            due_date = _optional_text(entry.get("due_date"))
        except ValueError as exc:
            raise ValueError(f"the optional fields are invalid: {exc}") from exc
        return ActionItem(id=item_id, description=description,
                          assignee=assignee, due_date=due_date,
                          evidence=tuple(evidence))

    question = _require_text(entry.get("question"))
    if question is None:
        raise ValueError("the element requires a non-empty 'question'")
    return OpenQuestion(id=item_id, question=question,
                        evidence=tuple(evidence))


def validate_analysis(payload: Dict[str, Any], transcript: str) -> AnalysisResult:
    """Validate the answer of the model against the transcription.

    Structural problems that affect the whole payload (a missing key, a
    collection that is not a list, an unsupported schema version) raise an
    ``AnalysisError``. Problems that affect a single element reject that element
    and are reported in the returned result.

    Parameters:
        payload: The parsed answer of the model.
        transcript: The transcription the answer refers to.

    Returns:
        AnalysisResult: The validated analysis and the problems found.

    Raises:
        AnalysisError: If the payload does not follow the expected schema.
    """
    if not isinstance(payload, dict):
        raise AnalysisError("The answer of the model is not a JSON object.")

    missing = [key for key in ("schema_version",) + COLLECTIONS if key not in payload]
    if missing:
        raise AnalysisError(
            "The answer of the model is missing required fields: "
            + ", ".join(missing) + "."
        )

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or schema_version.strip() != ANALYSIS_SCHEMA_VERSION:
        raise AnalysisError(
            f"Unsupported schema_version {schema_version!r}: the analysis requires "
            f"'{ANALYSIS_SCHEMA_VERSION}'."
        )

    normalized, segments = _index_transcript(transcript)
    has_timestamps = bool(segments)
    problems: List[ValidationProblem] = []

    topics = _validate_elements(COLLECTION_TOPICS, payload[COLLECTION_TOPICS],
                                normalized, segments, has_timestamps, problems)
    decisions = _validate_elements(COLLECTION_DECISIONS, payload[COLLECTION_DECISIONS],
                                   normalized, segments, has_timestamps, problems)
    action_items = _validate_elements(COLLECTION_ACTION_ITEMS,
                                      payload[COLLECTION_ACTION_ITEMS], normalized,
                                      segments, has_timestamps, problems)
    open_questions = _validate_elements(COLLECTION_OPEN_QUESTIONS,
                                        payload[COLLECTION_OPEN_QUESTIONS], normalized,
                                        segments, has_timestamps, problems)

    analysis = MeetingAnalysis(
        topics=tuple(topics),
        decisions=tuple(decisions),
        action_items=tuple(action_items),
        open_questions=tuple(open_questions),
        schema_version=ANALYSIS_SCHEMA_VERSION,
    )
    return AnalysisResult(analysis=analysis, problems=problems)


def analyze_transcript(transcript: str, config: larm.LarmConfig) -> AnalysisResult:
    """Request the analysis of a transcription and validate the answer.

    Parameters:
        transcript: Full content of the transcription.
        config: LARM connection settings.

    Returns:
        AnalysisResult: The validated analysis and the problems found.
    """
    raw = request_analysis(transcript, config)
    payload = parse_analysis_payload(raw)
    return validate_analysis(payload, transcript)


# =====================================================================================
# Export
# =====================================================================================
def render_json(analysis: MeetingAnalysis) -> str:
    """Return the analysis as an indented JSON document."""
    return json.dumps(analysis.to_dict(), ensure_ascii=False, indent=2) + "\n"


def format_timestamp(seconds: Optional[float]) -> str:
    """Return a time mark formatted as ``mm:ss.ss``."""
    if seconds is None:
        return "-"
    minutes, remainder = divmod(float(seconds), 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def _render_evidence(evidence: Sequence[Evidence], indent: str = "") -> List[str]:
    """Return the Markdown lines that list the evidence of an element."""
    lines = []
    for item in evidence:
        if item.start_seconds is None or item.end_seconds is None:
            lines.append(f"{indent}- «{item.quote}» (sin marca de tiempo)")
        else:
            lines.append(
                f"{indent}- «{item.quote}» ({format_timestamp(item.start_seconds)} – "
                f"{format_timestamp(item.end_seconds)})"
            )
    return lines


def render_markdown(analysis: MeetingAnalysis, transcript_path: Optional[str] = None,
                    model: Optional[str] = None,
                    generated_at: Optional[datetime] = None) -> str:
    """Return the analysis as a readable Markdown document."""
    stamp = (generated_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = ["# Análisis de reunión", ""]
    if transcript_path:
        lines.append(f"**Transcripción:** `{transcript_path}`  ")
    if model:
        lines.append(f"**Modelo:** `{model}`  ")
    lines.append(f"**Generado:** {stamp}  ")
    lines.append(f"**Esquema:** {analysis.schema_version}")
    lines.append("")

    lines.append("## Temas principales (Topics)")
    lines.append("")
    if analysis.topics:
        for topic in analysis.topics:
            lines.append(f"### {topic.id} — {topic.title}")
            lines.append("")
            lines.append(topic.description)
            lines.append("")
            lines.extend(_render_evidence(topic.evidence))
            lines.append("")
    else:
        lines.append("_No se identificaron temas._")
        lines.append("")

    lines.append("## Decisiones (Decisions)")
    lines.append("")
    if analysis.decisions:
        for decision in analysis.decisions:
            lines.append(f"### {decision.id}")
            lines.append("")
            lines.append(decision.description)
            lines.append("")
            lines.extend(_render_evidence(decision.evidence))
            lines.append("")
    else:
        lines.append("_No se identificaron decisiones confirmadas._")
        lines.append("")

    lines.append("## Tareas y responsables (Action items)")
    lines.append("")
    if analysis.action_items:
        for item in analysis.action_items:
            lines.append(f"### {item.id}")
            lines.append("")
            lines.append(item.description)
            lines.append("")
            lines.append(f"- **Responsable:** {item.assignee or 'no indicado'}")
            lines.append(f"- **Fecha límite:** {item.due_date or 'no indicada'}")
            lines.append("- **Evidencia:**")
            lines.extend(_render_evidence(item.evidence, indent="  "))
            lines.append("")
    else:
        lines.append("_No se identificaron tareas._")
        lines.append("")

    lines.append("## Preguntas pendientes (Open questions)")
    lines.append("")
    if analysis.open_questions:
        for question in analysis.open_questions:
            lines.append(f"### {question.id}")
            lines.append("")
            lines.append(question.question)
            lines.append("")
            lines.extend(_render_evidence(question.evidence))
            lines.append("")
    else:
        lines.append("_No se identificaron preguntas pendientes._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def target_paths(output_dir: str, base_name: str, output_format: str) -> List[str]:
    """Return the files that the given format writes."""
    paths = []
    if output_format in ("json", "both"):
        paths.append(os.path.join(output_dir, f"{base_name}.json"))
    if output_format in ("markdown", "both"):
        paths.append(os.path.join(output_dir, f"{base_name}.md"))
    return paths


def check_overwrite(paths: Sequence[str]) -> List[str]:
    """Return the files that already exist and would be replaced."""
    return [path for path in paths if os.path.exists(path)]


def _write_text(path: str, content: str) -> None:
    """Write a file atomically, replacing the destination only when it is ready."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".analysis-",
                                            suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def write_analysis(analysis: MeetingAnalysis, output_dir: str, output_format: str,
                   base_name: str, overwrite: bool = False,
                   transcript_path: Optional[str] = None,
                   model: Optional[str] = None,
                   generated_at: Optional[datetime] = None) -> List[str]:
    """Write the analysis as JSON and/or Markdown.

    Parameters:
        analysis: The validated analysis.
        output_dir: Directory where the files are written.
        output_format: One of ``json``, ``markdown`` or ``both``.
        base_name: Base name of the generated files.
        overwrite: Whether existing files may be replaced.
        transcript_path: Source transcription, only used by the Markdown header.
        model: Alias of the model used, only used by the Markdown header.
        generated_at: Timestamp written in the Markdown header.

    Returns:
        list: The paths of the written files.

    Raises:
        AnalysisError: If the format is unknown, a file already exists and
            ``overwrite`` is False, or the file cannot be written.
    """
    if output_format not in OUTPUT_FORMATS:
        raise AnalysisError(
            f"Unknown analysis format '{output_format}': use one of "
            + ", ".join(OUTPUT_FORMATS) + "."
        )

    paths = target_paths(output_dir, base_name, output_format)
    conflicts = check_overwrite(paths)
    if conflicts and not overwrite:
        raise AnalysisError(
            "The analysis files already exist and will not be replaced: "
            + ", ".join(conflicts)
            + ". Review them and rerun the command with --overwrite to replace them."
        )

    written: List[str] = []
    for path in paths:
        if path.endswith(".json"):
            content = render_json(analysis)
        else:
            content = render_markdown(analysis, transcript_path=transcript_path,
                                      model=model, generated_at=generated_at)
        try:
            _write_text(path, content)
        except OSError as exc:
            raise AnalysisError(f"Could not write the analysis file '{path}': {exc}") from exc
        written.append(path)
    return written

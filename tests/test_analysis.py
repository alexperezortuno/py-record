"""Unit tests for the structured analysis of transcriptions (REC-003).

All the tests mock the OpenAI compatible client, so no LARM instance is required
to run the suite.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import httpx
import openai

from meeting import analysis
from meeting import cli
from meeting import larm

REQUEST_URL = "http://127.0.0.1:8090/v1/chat/completions"

TRANSCRIPT = (
    "[0.00 --> 3.20] Ana: Vamos a revisar el despliegue de mañana.\n"
    "[3.20 --> 7.10] Luis: He terminado la migración de la base de datos.\n"
    "[7.10 --> 12.00] Ana: Entonces decidimos usar el plan B.\n"
    "[12.00 --> 18.30] Luis: ¿Quién actualiza la documentación del módulo?\n"
    "[18.30 --> 24.00] Ana: Luis actualiza la documentación antes del viernes.\n"
)


# --------------------------------------------------------------------------------------
# Test doubles
# --------------------------------------------------------------------------------------
class FakeBackend:
    """Records the requests sent to LARM and replays a canned answer."""

    def __init__(self):
        self.requests = []
        self.client_kwargs = None
        self.response = None
        self.error = None

    def openai_factory(self, **kwargs):
        self.client_kwargs = kwargs
        return FakeClient(self)


class FakeClient:
    def __init__(self, backend):
        self.backend = backend
        self.chat = FakeChat(backend)


class FakeChat:
    def __init__(self, backend):
        self.backend = backend
        self.completions = FakeCompletions(backend)


class FakeCompletions:
    def __init__(self, backend):
        self.backend = backend

    def create(self, **kwargs):
        self.backend.requests.append(kwargs)
        if self.backend.error is not None:
            raise self.backend.error
        return self.backend.response


def build_response(content):
    message = types.SimpleNamespace(role="assistant", content=content)
    choice = types.SimpleNamespace(index=0, message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], model="fake-model")


@contextlib.contextmanager
def larm_env(**values):
    """Run the block with a clean set of LARM_* environment variables."""
    saved = {key: os.environ.pop(key) for key in list(os.environ) if key.startswith("LARM_")}
    os.environ.update(values)
    try:
        yield
    finally:
        for key in [key for key in os.environ if key.startswith("LARM_")]:
            del os.environ[key]
        os.environ.update(saved)


def analyze_args(**overrides):
    values = {
        "transcript": "reunion.txt",
        "provider": "larm",
        "larm_model": None,
        "format": "both",
        "output": None,
        "overwrite": False,
        "env": None,
    }
    values.update(overrides)
    return Namespace(**values)


def quote(text, start=None, end=None):
    """Build an evidence entry."""
    return {"quote": text, "start_seconds": start, "end_seconds": end}


def valid_payload(**overrides):
    """Build a payload that passes the validation."""
    payload = {
        "schema_version": "1.0",
        "topics": [
            {
                "id": "T1",
                "title": "Despliegue de mañana",
                "description": "Se revisó el despliegue de mañana.",
                "evidence": [quote("Vamos a revisar el despliegue de mañana", 0.0, 3.2)],
            }
        ],
        "decisions": [
            {
                "id": "D1",
                "description": "Se decidió usar el plan B.",
                "evidence": [quote("decidimos usar el plan B", 7.1, 12.0)],
            }
        ],
        "action_items": [
            {
                "id": "A1",
                "description": "Actualizar la documentación del módulo.",
                "assignee": "Luis",
                "due_date": None,
                "evidence": [quote("Luis actualiza la documentación antes del viernes",
                                   18.3, 24.0)],
            }
        ],
        "open_questions": [
            {
                "id": "Q1",
                "question": "¿Quién actualiza la documentación del módulo?",
                "evidence": [quote("¿Quién actualiza la documentación del módulo?",
                                   12.0, 18.3)],
            }
        ],
    }
    payload.update(overrides)
    return payload


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.patcher = patch("openai.OpenAI", self.backend.openai_factory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    # ---------------------------------------------------------------- helpers
    def write_transcript(self, name="reunion.txt", text=TRANSCRIPT):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def respond_with(self, payload):
        self.backend.response = build_response(json.dumps(payload, ensure_ascii=False))

    # --------------------------------------------------------------- TEST-01
    def test_analyze_command_is_available(self):
        with patch.object(sys, "argv", ["meeting", "analyze", "--help"]):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 0)

        with patch.object(cli, "action_analyze") as action:
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", "reunion.txt"]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(action.call_args.args[0].transcript, "reunion.txt")

    def test_analyze_command_is_registered_with_the_documented_options(self):
        buffer = io.StringIO()
        with patch.object(sys, "argv", ["meeting", "analyze", "--help"]):
            with contextlib.redirect_stdout(buffer):
                with self.assertRaises(SystemExit):
                    cli.main()
        usage = buffer.getvalue()
        for option in ("--transcript", "--provider", "--larm-model", "--format",
                       "--output", "--overwrite"):
            self.assertIn(option, usage)
        self.assertIn("{larm}", usage)
        self.assertIn("{json,markdown,both}", usage)

    # --------------------------------------------------------------- TEST-02
    def test_existing_transcription_is_read(self):
        path = self.write_transcript()
        text = analysis.read_transcript(path)
        self.assertEqual(text, TRANSCRIPT)

    def test_existing_transcription_is_read_by_the_command(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(self.backend.requests[0]["messages"][1]["content"], TRANSCRIPT)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "reunion_analysis.json")))

    def test_transcription_is_not_modified(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        before = os.stat(path)
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)
        after = os.stat(path)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), TRANSCRIPT)
        self.assertEqual(before.st_size, after.st_size)

    # --------------------------------------------------------------- TEST-03
    def test_larm_is_the_default_provider_and_the_alias_is_used(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias-env"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(self.backend.requests[0]["model"], "alias-env")
        self.assertEqual(self.backend.client_kwargs["base_url"],
                         larm.LARM_DEFAULT_BASE_URL)

    def test_command_line_alias_overrides_the_environment_model(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias-env"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--larm-model", "alias-cli"]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(self.backend.requests[0]["model"], "alias-cli")

    def test_missing_model_is_reported_without_calling_larm(self):
        path = self.write_transcript()
        with larm_env():
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)
        self.assertEqual(self.backend.requests, [])

    def test_only_larm_is_accepted_as_provider(self):
        path = self.write_transcript()
        with patch.object(sys, "argv",
                          ["meeting", "analyze", "--transcript", path,
                           "--provider", "openai"]):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 2)

    # --------------------------------------------------------------- TEST-04
    def test_request_to_the_model_is_correct(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)

        self.assertEqual(len(self.backend.requests), 1)
        request = self.backend.requests[0]
        self.assertEqual(request["model"], "alias")
        self.assertEqual([message["role"] for message in request["messages"]],
                         ["system", "user"])
        self.assertEqual(request["messages"][1]["content"], TRANSCRIPT)
        self.assertNotIn("temperature", request)
        self.assertNotIn("response_format", request)
        self.assertNotIn("max_tokens", request)

        system_prompt = request["messages"][0]["content"]
        self.assertIn("ÚNICA fuente de información autorizada", system_prompt)
        self.assertIn("schema_version", system_prompt)
        self.assertIn("open_questions", system_prompt)
        self.assertNotEqual(system_prompt, larm.build_system_prompt("en"))

    def test_the_transcription_is_never_truncated(self):
        self.respond_with(valid_payload())
        long_transcript = TRANSCRIPT * 200
        path = self.write_transcript(text=long_transcript)
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(self.backend.requests[0]["messages"][1]["content"],
                         long_transcript)

    # --------------------------------------------------------------- TEST-05
    def test_valid_json_is_accepted(self):
        payload = valid_payload()
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.analysis.count(), 4)

    def test_json_inside_a_code_block_is_accepted(self):
        raw = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```"
        parsed = analysis.parse_analysis_payload(raw)
        self.assertEqual(parsed["schema_version"], "1.0")

    def test_json_with_surrounding_text_is_accepted(self):
        raw = "Aquí tienes el análisis:\n" + json.dumps(valid_payload(),
                                                        ensure_ascii=False)
        parsed = analysis.parse_analysis_payload(raw)
        self.assertEqual(parsed["schema_version"], "1.0")

    # --------------------------------------------------------------- TEST-06
    def test_invalid_json_is_reported(self):
        with self.assertRaises(analysis.AnalysisError) as raised:
            analysis.parse_analysis_payload("esto no es json")
        self.assertIn("not valid JSON", str(raised.exception))

    def test_invalid_json_finishes_with_a_controlled_error(self):
        self.backend.response = build_response("{not json")
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)
        self.assertFalse(os.path.exists(os.path.join(self.root, "reunion_analysis.json")))

    def test_empty_answer_is_reported(self):
        self.backend.response = build_response("")
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)

    # --------------------------------------------------------------- TEST-07
    def test_missing_required_field_is_reported(self):
        payload = valid_payload()
        del payload["open_questions"]
        with self.assertRaises(analysis.AnalysisError) as raised:
            analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertIn("missing required fields", str(raised.exception))

    def test_wrong_type_on_a_collection_is_reported(self):
        payload = valid_payload()
        payload["topics"] = {"not": "a list"}
        with self.assertRaises(analysis.AnalysisError) as raised:
            analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertIn("must be a list", str(raised.exception))

    def test_wrong_type_on_a_field_rejects_the_element(self):
        payload = valid_payload()
        payload["decisions"][0]["description"] = 42
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)
        self.assertIn("description", result.problems[0].message)

    def test_wrong_type_on_an_optional_field_rejects_the_element(self):
        payload = valid_payload()
        payload["action_items"][0]["assignee"] = ["Luis"]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.action_items, ())
        self.assertEqual(len(result.problems), 1)

    def test_unsupported_schema_version_is_reported(self):
        payload = valid_payload()
        payload["schema_version"] = "2.0"
        with self.assertRaises(analysis.AnalysisError) as raised:
            analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertIn("schema_version", str(raised.exception))

    def test_duplicated_identifiers_are_rejected(self):
        payload = valid_payload()
        payload["decisions"].append({
            "id": "D1",
            "description": "Otra decisión.",
            "evidence": [quote("decidimos usar el plan B", 7.1, 12.0)],
        })
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(len(result.analysis.decisions), 1)
        self.assertEqual(len(result.problems), 1)
        self.assertIn("duplicated", result.problems[0].message)

    # --------------------------------------------------------------- TEST-08
    def test_topics_are_extracted(self):
        result = analysis.validate_analysis(valid_payload(), TRANSCRIPT)
        self.assertEqual(len(result.analysis.topics), 1)
        topic = result.analysis.topics[0]
        self.assertEqual(topic.id, "T1")
        self.assertEqual(topic.title, "Despliegue de mañana")
        self.assertEqual(topic.description, "Se revisó el despliegue de mañana.")
        self.assertEqual(len(topic.evidence), 1)
        self.assertEqual(topic.evidence[0].quote,
                         "Vamos a revisar el despliegue de mañana")

    # --------------------------------------------------------------- TEST-09
    def test_decisions_are_extracted(self):
        result = analysis.validate_analysis(valid_payload(), TRANSCRIPT)
        self.assertEqual(len(result.analysis.decisions), 1)
        decision = result.analysis.decisions[0]
        self.assertEqual(decision.id, "D1")
        self.assertEqual(decision.description, "Se decidió usar el plan B.")

    # -------------------------------------------------------------- TEST-10
    def test_action_items_and_assignees_are_extracted(self):
        result = analysis.validate_analysis(valid_payload(), TRANSCRIPT)
        self.assertEqual(len(result.analysis.action_items), 1)
        item = result.analysis.action_items[0]
        self.assertEqual(item.id, "A1")
        self.assertEqual(item.description, "Actualizar la documentación del módulo.")
        self.assertEqual(item.assignee, "Luis")
        self.assertEqual(item.due_date, None)

    # -------------------------------------------------------------- TEST-11
    def test_optional_fields_are_null_when_they_are_not_mentioned(self):
        payload = valid_payload()
        payload["action_items"][0]["assignee"] = None
        payload["action_items"][0]["due_date"] = None
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        item = result.analysis.action_items[0]
        self.assertIsNone(item.assignee)
        self.assertIsNone(item.due_date)

    def test_missing_optional_fields_are_read_as_null(self):
        payload = valid_payload()
        del payload["action_items"][0]["assignee"]
        del payload["action_items"][0]["due_date"]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        item = result.analysis.action_items[0]
        self.assertIsNone(item.assignee)
        self.assertIsNone(item.due_date)

    def test_empty_categories_return_an_empty_list(self):
        payload = valid_payload(topics=[], decisions=[], action_items=[],
                                open_questions=[])
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.topics, ())
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(result.analysis.action_items, ())
        self.assertEqual(result.analysis.open_questions, ())

    # -------------------------------------------------------------- TEST-12
    def test_open_questions_are_extracted(self):
        result = analysis.validate_analysis(valid_payload(), TRANSCRIPT)
        self.assertEqual(len(result.analysis.open_questions), 1)
        question = result.analysis.open_questions[0]
        self.assertEqual(question.id, "Q1")
        self.assertIn("documentación", question.question)

    # -------------------------------------------------------------- TEST-13
    def test_literal_quotes_are_validated(self):
        payload = valid_payload()
        payload["decisions"][0]["evidence"] = [quote("se decidió usar el plan B")]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)
        self.assertIn("does not exist literally", result.problems[0].message)

    def test_a_valid_quote_with_different_spacing_is_accepted(self):
        payload = valid_payload()
        payload["decisions"][0]["evidence"] = [
            quote("Entonces   decidimos usar el plan B.")]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(len(result.analysis.decisions), 1)

    # -------------------------------------------------------------- TEST-14
    def test_timestamps_are_validated_against_the_source_segment(self):
        payload = valid_payload()
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 7.1, 12.0)]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(len(result.analysis.decisions), 1)
        evidence = result.analysis.decisions[0].evidence[0]
        self.assertAlmostEqual(evidence.start_seconds, 7.1)
        self.assertAlmostEqual(evidence.end_seconds, 12.0)

    def test_wrong_timestamps_reject_the_element(self):
        payload = valid_payload()
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 120.0, 130.0)]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)
        self.assertIn("does not match the source segment", result.problems[0].message)

    def test_inverted_timestamps_reject_the_element(self):
        payload = valid_payload()
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 12.0, 7.1)]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)

    def test_timestamps_are_not_invented_when_the_transcript_has_none(self):
        plain = ("Ana: Vamos a revisar el despliegue de mañana.\n"
                 "Luis: Entonces decidimos usar el plan B.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 7.1, 12.0)]
        result = analysis.validate_analysis(payload, plain)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)
        self.assertIn("does not provide time marks", result.problems[0].message)

    def test_transcript_without_timestamps_accepts_evidence_without_timestamps(self):
        plain = ("Ana: Vamos a revisar el despliegue de mañana.\n"
                 "Luis: Entonces decidimos usar el plan B.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B")]
        result = analysis.validate_analysis(payload, plain)
        self.assertEqual(len(result.analysis.decisions), 1)
        self.assertIsNone(result.analysis.decisions[0].evidence[0].start_seconds)

    def test_a_quote_repeated_in_the_transcript_accepts_the_matching_occurrence(self):
        repeated = ("[0.00 --> 3.20] Ana: Revisamos el despliegue.\n"
                    "[3.20 --> 7.10] Luis: Revisamos el despliegue.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["description"] = "Se revisó el despliegue."
        payload["decisions"][0]["evidence"] = [quote("Revisamos el despliegue", 3.2, 7.1)]
        result = analysis.validate_analysis(payload, repeated)
        self.assertEqual(len(result.analysis.decisions), 1)

    def test_a_time_mark_written_by_the_model_is_accepted_when_the_segment_confirms_it(self):
        gemini_like = ("**Hablante 1** [04:48]: entonces decidimos usar el plan B.\n"
                       "**Hablante 2** [05:20]: vale, lo revisamos mañana.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 4.48, 4.48)]
        result = analysis.validate_analysis(payload, gemini_like)
        self.assertEqual(len(result.analysis.decisions), 1)
        evidence = result.analysis.decisions[0].evidence[0]
        self.assertAlmostEqual(evidence.start_seconds, 288.0)
        self.assertAlmostEqual(evidence.end_seconds, 288.0)

    def test_a_time_mark_that_no_segment_confirms_is_rejected(self):
        gemini_like = ("**Hablante 1** [04:48]: entonces decidimos usar el plan B.\n"
                       "**Hablante 2** [05:20]: vale, lo revisamos mañana.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 9.99, 9.99)]
        result = analysis.validate_analysis(payload, gemini_like)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)

    def test_missing_timestamps_are_anchored_in_the_source_segment(self):
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B")]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(len(result.analysis.decisions), 1)
        evidence = result.analysis.decisions[0].evidence[0]
        self.assertAlmostEqual(evidence.start_seconds, 7.1)
        self.assertAlmostEqual(evidence.end_seconds, 12.0)

    def test_missing_timestamps_stay_null_without_time_marks(self):
        plain = ("Ana: Vamos a revisar el despliegue de mañana.\n"
                 "Luis: Entonces decidimos usar el plan B.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B")]
        result = analysis.validate_analysis(payload, plain)
        evidence = result.analysis.decisions[0].evidence[0]
        self.assertIsNone(evidence.start_seconds)
        self.assertIsNone(evidence.end_seconds)

    def test_a_quote_that_only_differs_in_the_case_is_accepted(self):
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("Entonces decidimos usar el plan B.")]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(len(result.analysis.decisions), 1)
        evidence = result.analysis.decisions[0].evidence[0]
        self.assertEqual(evidence.quote, "Entonces decidimos usar el plan B.")

    def test_the_stored_quote_is_always_the_fragment_of_the_transcript(self):
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("Entonces decidimos usar el plan B.")]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        evidence = result.analysis.decisions[0].evidence[0]
        self.assertIn(evidence.quote, " ".join(TRANSCRIPT.split()))

    def test_transcript_with_single_time_marks_validates_the_timestamps(self):
        gemini_like = ("**Hablante 1** [04:48]: entonces decidimos usar el plan B.\n"
                       "**Hablante 2** [05:20]: vale, lo revisamos mañana.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 288.0, 295.0)]
        result = analysis.validate_analysis(payload, gemini_like)
        self.assertEqual(len(result.analysis.decisions), 1)
        self.assertAlmostEqual(result.analysis.decisions[0].evidence[0].start_seconds,
                               288.0)

    def test_transcript_with_single_time_marks_rejects_wrong_timestamps(self):
        gemini_like = ("**Hablante 1** [04:48]: entonces decidimos usar el plan B.\n"
                       "**Hablante 2** [05:20]: vale, lo revisamos mañana.\n")
        payload = valid_payload(topics=[], action_items=[], open_questions=[])
        payload["decisions"][0]["evidence"] = [quote("decidimos usar el plan B", 1200.0, 1210.0)]
        result = analysis.validate_analysis(payload, gemini_like)
        self.assertEqual(result.analysis.decisions, ())
        self.assertEqual(len(result.problems), 1)

    # -------------------------------------------------------------- TEST-15
    def test_invented_evidence_is_rejected(self):
        payload = valid_payload()
        payload["topics"][0]["evidence"] = [
            quote("El equipo decidió cancelar el proyecto la semana pasada.")
        ]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.topics, ())
        self.assertEqual(len(result.problems), 1)

    def test_an_element_without_evidence_is_rejected(self):
        payload = valid_payload()
        payload["topics"][0]["evidence"] = []
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.topics, ())
        self.assertEqual(len(result.problems), 1)

    def test_a_rejected_element_does_not_remove_the_valid_ones(self):
        payload = valid_payload()
        payload["topics"][0]["evidence"] = [quote("una cita que no existe")]
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        self.assertEqual(result.analysis.topics, ())
        self.assertEqual(len(result.analysis.decisions), 1)
        self.assertEqual(len(result.analysis.action_items), 1)
        self.assertEqual(len(result.analysis.open_questions), 1)

    # -------------------------------------------------------------- TEST-16
    def test_json_export(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--format", "json"]):
                self.assertEqual(cli.main(), 0)

        exported = os.path.join(self.root, "reunion_analysis.json")
        self.assertTrue(os.path.isfile(exported))
        self.assertFalse(os.path.exists(os.path.join(self.root, "reunion_analysis.md")))
        with open(exported, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual(data["schema_version"], "1.0")
        self.assertEqual(len(data["topics"]), 1)
        self.assertEqual(len(data["decisions"]), 1)
        self.assertEqual(len(data["action_items"]), 1)
        self.assertEqual(len(data["open_questions"]), 1)
        self.assertEqual(data["action_items"][0]["assignee"], "Luis")
        self.assertIsNone(data["action_items"][0]["due_date"])
        self.assertEqual(data["decisions"][0]["evidence"][0]["quote"],
                         "decidimos usar el plan B")

    # -------------------------------------------------------------- TEST-17
    def test_markdown_export(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--format", "markdown"]):
                self.assertEqual(cli.main(), 0)

        exported = os.path.join(self.root, "reunion_analysis.md")
        self.assertTrue(os.path.isfile(exported))
        self.assertFalse(os.path.exists(os.path.join(self.root, "reunion_analysis.json")))
        with open(exported, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("## Temas principales", content)
        self.assertIn("## Decisiones", content)
        self.assertIn("## Tareas y responsables", content)
        self.assertIn("## Preguntas pendientes", content)
        self.assertIn("Despliegue de mañana", content)
        self.assertIn("Luis", content)
        self.assertIn("decidimos usar el plan B", content)
        self.assertIn("00:07.10", content)

    # -------------------------------------------------------------- TEST-18
    def test_both_formats_are_exported(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--format", "both"]):
                self.assertEqual(cli.main(), 0)

        self.assertTrue(os.path.isfile(os.path.join(self.root, "reunion_analysis.json")))
        self.assertTrue(os.path.isfile(os.path.join(self.root, "reunion_analysis.md")))

    def test_output_directory_is_used(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        destination = os.path.join(self.root, "analysis")
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--output", destination]):
                self.assertEqual(cli.main(), 0)
        self.assertTrue(os.path.isfile(os.path.join(destination, "reunion_analysis.json")))
        self.assertTrue(os.path.isfile(os.path.join(destination, "reunion_analysis.md")))

    # -------------------------------------------------------------- TEST-19
    def test_existing_files_are_not_overwritten(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        existing = os.path.join(self.root, "reunion_analysis.json")
        with open(existing, "w", encoding="utf-8") as handle:
            handle.write("contenido previo")

        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)

        with open(existing, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "contenido previo")
        self.assertEqual(self.backend.requests, [])

    def test_overwrite_requires_an_explicit_action(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        existing = os.path.join(self.root, "reunion_analysis.json")
        with open(existing, "w", encoding="utf-8") as handle:
            handle.write("contenido previo")

        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--overwrite"]):
                self.assertEqual(cli.main(), 0)

        with open(existing, encoding="utf-8") as handle:
            self.assertNotEqual(handle.read(), "contenido previo")

    # -------------------------------------------------------------- TEST-20
    def test_missing_transcript_is_reported(self):
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript",
                               os.path.join(self.root, "missing.txt")]):
                self.assertEqual(cli.main(), 1)
        self.assertEqual(self.backend.requests, [])

    def test_empty_transcript_is_reported(self):
        path = self.write_transcript(text="   \n")
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)
        self.assertEqual(self.backend.requests, [])

    def test_invalid_encoding_is_reported(self):
        path = os.path.join(self.root, "reunion.txt")
        with open(path, "wb") as handle:
            handle.write(b"\xff\xfe\x00binario no es texto")
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)
        self.assertEqual(self.backend.requests, [])
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), b"\xff\xfe\x00binario no es texto")

    # -------------------------------------------------------------- TEST-21
    def test_larm_connection_error_is_reported(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APIConnectionError(request=request)
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)
        self.assertFalse(os.path.exists(os.path.join(self.root, "reunion_analysis.json")))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), TRANSCRIPT)

    def test_larm_connection_error_is_translated(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APIConnectionError(request=request)
        with larm_env(LARM_MODEL="alias"):
            config = larm.resolve_config(None)
            with self.assertRaises(analysis.AnalysisError) as raised:
                analysis.request_analysis(TRANSCRIPT, config)
        self.assertIn("Could not connect to LARM", str(raised.exception))

    # -------------------------------------------------------------- TEST-22
    def test_timeout_is_reported(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APITimeoutError(request=request)
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias", LARM_TIMEOUT="30"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 1)
        self.assertFalse(os.path.exists(os.path.join(self.root, "reunion_analysis.json")))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), TRANSCRIPT)

    def test_timeout_message_mentions_the_limit(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APITimeoutError(request=request)
        with larm_env(LARM_MODEL="alias", LARM_TIMEOUT="30"):
            config = larm.resolve_config(None)
            with self.assertRaises(analysis.AnalysisError) as raised:
                analysis.request_analysis(TRANSCRIPT, config)
        self.assertIn("timed out after 30s", str(raised.exception))

    # -------------------------------------------------------------- TEST-23
    def test_interruption_is_reported_with_the_interrupt_exit_code(self):
        self.backend.error = KeyboardInterrupt()
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 130)
        self.assertFalse(os.path.exists(os.path.join(self.root, "reunion_analysis.json")))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), TRANSCRIPT)

    def test_interruption_does_not_delete_previous_results(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)
        exported = os.path.join(self.root, "reunion_analysis.json")
        with open(exported, encoding="utf-8") as handle:
            previous = handle.read()

        self.backend.error = KeyboardInterrupt()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path,
                               "--overwrite"]):
                self.assertEqual(cli.main(), 130)
        with open(exported, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), previous)

    # -------------------------------------------------------------- TEST-24
    def test_previous_commands_keep_their_behavior(self):
        with patch.object(cli, "action_transcript") as action:
            with patch.object(sys, "argv",
                              ["meeting", "transcript", "--model", "larm",
                               "--audio", "x.mp3"]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(action.call_args.args[0].model, "larm")

        with patch.object(cli, "action_process") as action:
            with patch.object(sys, "argv", ["meeting", "process"]):
                self.assertEqual(cli.main(), 0)
        self.assertTrue(action.called)

        with patch.object(cli, "action_record") as action:
            with patch.object(sys, "argv", ["meeting", "record"]):
                self.assertEqual(cli.main(), 0)
        self.assertTrue(action.called)

        with patch.object(cli, "action_diarize") as action:
            with patch.object(sys, "argv", ["meeting", "diarize", "--audio", "x.mp3"]):
                self.assertEqual(cli.main(), 0)
        self.assertTrue(action.called)

    def test_analyze_does_not_touch_the_database(self):
        self.respond_with(valid_payload())
        path = self.write_transcript()
        with larm_env(LARM_MODEL="alias"):
            with patch.object(sys, "argv",
                              ["meeting", "analyze", "--transcript", path]):
                self.assertEqual(cli.main(), 0)
        self.assertFalse(os.path.exists(os.path.join(self.root, "meetings.db")))

    # ---------------------------------------------------------------- extras
    def test_markdown_renders_the_missing_values(self):
        payload = valid_payload()
        payload["action_items"][0]["assignee"] = None
        payload["action_items"][0]["due_date"] = None
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        content = analysis.render_markdown(result.analysis)
        self.assertIn("no indicado", content)
        self.assertIn("no indicada", content)

    def test_markdown_renders_empty_categories(self):
        payload = valid_payload(topics=[], decisions=[], action_items=[],
                                open_questions=[])
        result = analysis.validate_analysis(payload, TRANSCRIPT)
        content = analysis.render_markdown(result.analysis)
        self.assertIn("_No se identificaron temas._", content)
        self.assertIn("_No se identificaron decisiones confirmadas._", content)
        self.assertIn("_No se identificaron tareas._", content)
        self.assertIn("_No se identificaron preguntas pendientes._", content)

    def test_source_segments_are_indexed(self):
        normalized, segments = analysis._index_transcript(TRANSCRIPT)
        self.assertEqual(len(segments), 5)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 3.2)
        self.assertIn("Vamos a revisar el despliegue de mañana", normalized)


if __name__ == "__main__":
    unittest.main()

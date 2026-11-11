"""Unit tests for the LARM provider (REC-002).

All the tests mock the OpenAI compatible client, so no LARM instance is required.
"""

import contextlib
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import httpx
import openai

from meeting import cli
from meeting import larm

REQUEST_URL = "http://127.0.0.1:8090/v1/chat/completions"


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
        self.models = []

    def openai_factory(self, **kwargs):
        self.client_kwargs = kwargs
        return FakeClient(self)


class FakeClient:
    def __init__(self, backend):
        self.backend = backend
        self.chat = FakeChat(backend)
        self.models = FakeModels(backend)


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


class FakeModels:
    def __init__(self, backend):
        self.backend = backend

    def list(self):
        return types.SimpleNamespace(data=[types.SimpleNamespace(id=name)
                                          for name in self.backend.models])


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


def larm_args(**overrides):
    values = {
        "model": "larm",
        "lang": "es",
        "prompt": "You are an expert meeting summary assistant.",
        "larm_model": None,
        "whisper_model": None,
        "whisper_device": None,
        "export_md": True,
        "output": ".",
        "env": None,
    }
    values.update(overrides)
    return Namespace(**values)


def api_error(error_class, status_code, message):
    request = httpx.Request("POST", REQUEST_URL)
    response = httpx.Response(status_code, request=request)
    return error_class(message, response=response, body=None)


class LarmTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.patcher = patch("openai.OpenAI", self.backend.openai_factory)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name
        self.db_path = os.path.join(self.root, "meetings.db")
        self.db_patcher = patch.object(cli, "DB_PATH", self.db_path)
        self.db_patcher.start()
        self.addCleanup(self.db_patcher.stop)
        self.markdown_patcher = patch.object(cli, "MARKDOWN_DIR",
                                             os.path.join(self.root, "markdowns"))
        self.markdown_patcher.start()
        self.addCleanup(self.markdown_patcher.stop)
        # The CLI writes the exported markdown relative to the working directory:
        # the test runs inside the temporary folder to keep the repository clean.
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    # ---------------------------------------------------------------- helpers
    def write_transcription(self, name="meeting_transcription.txt", text="Contenido real."):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def database_rows(self):
        connection = sqlite3.connect(self.db_path)
        try:
            cursor = connection.execute(
                "SELECT transcript_name, transcript_path, summary_path, mode FROM meetings"
            )
            return cursor.fetchall()
        finally:
            connection.close()

    # --------------------------------------------------------------- TEST-01
    def test_larm_provider_is_selectable_from_the_cli(self):
        with patch.object(cli, "action_transcript") as action:
            with patch.object(sys, "argv",
                              ["meeting", "transcript", "--model", "larm", "--audio", "x.mp3"]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(action.call_args.args[0].model, "larm")

    def test_previous_providers_remain_selectable(self):
        for legacy in ("openai", "gemini", "local"):
            with patch.object(cli, "action_transcript") as action:
                with patch.object(sys, "argv",
                                  ["meeting", "transcript", "--mode", legacy, "--audio", "x.mp3"]):
                    self.assertEqual(cli.main(), 0)
            self.assertEqual(action.call_args.args[0].model, legacy)

    # --------------------------------------------------------------- TEST-02
    def test_default_url_and_configured_model(self):
        self.assertEqual(larm.LARM_DEFAULT_BASE_URL, "http://127.0.0.1:8090/v1")
        self.assertEqual(larm.LARM_DEFAULT_TIMEOUT, 600.0)
        self.assertEqual(larm.LARM_DEFAULT_API_KEY, "local")

        with larm_env(LARM_MODEL="local-alias"):
            config = larm.resolve_config(None)
        self.assertEqual(config.base_url, "http://127.0.0.1:8090/v1")
        self.assertEqual(config.model, "local-alias")
        self.assertEqual(config.timeout, 600.0)
        self.assertEqual(config.api_key, "local")

        with larm_env(LARM_BASE_URL="http://127.0.0.1:8090/v1/", LARM_MODEL="local-alias"):
            config = larm.resolve_config(None)
        self.assertEqual(config.base_url, "http://127.0.0.1:8090/v1")
        self.assertEqual(config.model, "local-alias")

    def test_client_uses_configured_url_model_and_timeout(self):
        self.backend.response = build_response("Resumen")
        path = self.write_transcription()
        with larm_env(LARM_BASE_URL="http://127.0.0.1:9000/v1", LARM_MODEL="alias-env",
                      LARM_TIMEOUT="42", LARM_API_KEY="local"):
            cli.summary_larm(path, larm_args())

        self.assertEqual(self.backend.client_kwargs["base_url"], "http://127.0.0.1:9000/v1")
        self.assertEqual(self.backend.client_kwargs["timeout"], 42.0)
        self.assertEqual(self.backend.client_kwargs["api_key"], "local")
        self.assertEqual(self.backend.requests[0]["model"], "alias-env")

    def test_command_line_model_overrides_environment_model(self):
        self.backend.response = build_response("Resumen")
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias-env"):
            cli.summary_larm(path, larm_args(larm_model="alias-cli"))
        self.assertEqual(self.backend.requests[0]["model"], "alias-cli")

    # --------------------------------------------------------------- TEST-03
    def test_chat_completion_request_is_valid(self):
        self.backend.response = build_response("Resumen")
        transcription = "Línea de transcripción.\n" * 3000
        path = self.write_transcription(text=transcription)

        with larm_env(LARM_MODEL="alias"):
            cli.summary_larm(path, larm_args())

        self.assertEqual(len(self.backend.requests), 1)
        request = self.backend.requests[0]
        self.assertEqual(request["model"], "alias")
        self.assertEqual([message["role"] for message in request["messages"]],
                         ["system", "user"])
        self.assertEqual(request["messages"][1]["content"], transcription)
        self.assertNotIn("temperature", request)
        self.assertNotIn("response_format", request)
        self.assertNotIn("max_tokens", request)
        self.assertTrue(request["messages"][0]["content"].strip())

    def test_system_prompt_is_meeting_oriented_and_in_spanish(self):
        prompt = larm.build_system_prompt("es")
        self.assertIn("resumir transcripciones de reuniones", prompt)
        self.assertIn("No inventes hechos, participantes, decisiones ni conclusiones", prompt)
        self.assertIn("No incluyas información que no aparezca en la transcripción", prompt)
        self.assertIn("Mantén los nombres técnicos, identificadores", prompt)
        self.assertIn("insuficiente, ambigua o incompleta", prompt)

    # --------------------------------------------------------------- TEST-04
    def test_summary_is_generated_and_saved(self):
        self.backend.response = build_response("## Resumen\n\nPunto principal.")
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias"):
            output = cli.summary_larm(path, larm_args())

        self.assertTrue(output.endswith("_summary_larm.txt"))
        with open(output, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "## Resumen\n\nPunto principal.")

    def test_transcript_flow_saves_summary_and_database_row(self):
        self.backend.response = build_response("Resumen generado")
        audio = os.path.join(self.root, "meeting.mp3")
        open(audio, "wb").close()
        transcript = self.write_transcription(name="meeting_transcription_local.txt")
        args = larm_args(audio=audio, output=self.root, larm_model="alias")

        with patch.object(cli, "transcript_local", return_value=transcript):
            cli.action_transcript(args)

        rows = self.database_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], "larm")
        self.assertTrue(rows[0][2].endswith("_summary_larm.txt"))
        self.assertTrue(os.path.isfile(rows[0][1]))
        self.assertTrue(os.path.isfile(rows[0][2]))
        markdown = os.path.join(self.root, "meeting.md")
        self.assertTrue(os.path.isfile(markdown))
        with open(markdown, encoding="utf-8") as handle:
            content = handle.read()
        self.assertIn("Resumen generado", content)
        self.assertIn("Contenido real.", content)

    # --------------------------------------------------------------- TEST-05
    def test_larm_not_available_is_reported_and_transcript_is_preserved(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APIConnectionError(request=request)
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("Could not connect to LARM", str(raised.exception))
        self.assertIn("127.0.0.1:8090", str(raised.exception))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Contenido real.")

    def test_connection_failure_keeps_the_transcription_row(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APIConnectionError(request=request)
        audio = os.path.join(self.root, "meeting.mp3")
        open(audio, "wb").close()
        transcript = self.write_transcription(name="meeting_transcription_local.txt")
        args = larm_args(audio=audio, output=self.root, larm_model="alias")

        with patch.object(cli, "transcript_local", return_value=transcript):
            with self.assertRaises(larm.LarmError):
                cli.action_transcript(args)

        rows = self.database_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], transcript)
        self.assertIsNone(rows[0][2])
        with open(transcript, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Contenido real.")

    # --------------------------------------------------------------- TEST-06
    def test_inference_timeout_is_reported(self):
        request = httpx.Request("POST", REQUEST_URL)
        self.backend.error = openai.APITimeoutError(request=request)
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias", LARM_TIMEOUT="30"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("timed out after 30s", str(raised.exception))
        self.assertIn("alias", str(raised.exception))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Contenido real.")

    def test_http_server_error_is_reported(self):
        self.backend.error = api_error(openai.APIStatusError, 500, "internal error")
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("HTTP 500", str(raised.exception))

    def test_context_window_overflow_is_reported(self):
        self.backend.error = api_error(
            openai.BadRequestError, 400, "This model's maximum context length is 8192 tokens"
        )
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("context window", str(raised.exception))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Contenido real.")

    # --------------------------------------------------------------- TEST-07
    def test_missing_model_is_reported_without_calling_larm(self):
        path = self.write_transcription()
        with larm_env():
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("No LARM model configured", str(raised.exception))
        self.assertEqual(self.backend.requests, [])

    def test_unknown_model_returns_a_clear_error(self):
        self.backend.error = api_error(openai.NotFoundError, 404, "model not found")
        path = self.write_transcription()
        with larm_env(LARM_MODEL="missing-alias"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("missing-alias", str(raised.exception))
        self.assertIn("not found", str(raised.exception))

    # --------------------------------------------------------------- TEST-08
    def test_empty_response_is_reported(self):
        self.backend.response = build_response("")
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("empty summary", str(raised.exception))
        self.assertFalse(os.path.exists(path.replace(".txt", "_summary_larm.txt")))

    def test_response_without_choices_is_reported(self):
        self.backend.response = types.SimpleNamespace(choices=[])
        path = self.write_transcription()
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(larm.LarmError) as raised:
                cli.summary_larm(path, larm_args())
        self.assertIn("no choices", str(raised.exception))

    # --------------------------------------------------------------- TEST-09
    def test_empty_transcription_is_rejected_without_calling_larm(self):
        path = self.write_transcription(text="   \n")
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(ValueError):
                cli.summary_larm(path, larm_args())
        self.assertEqual(self.backend.requests, [])

    def test_missing_transcription_is_reported(self):
        with larm_env(LARM_MODEL="alias"):
            with self.assertRaises(FileNotFoundError):
                cli.summary_larm(os.path.join(self.root, "missing.txt"), larm_args())

    # --------------------------------------------------------------- TEST-10
    def test_openai_provider_still_generates_a_summary(self):
        self.backend.response = build_response("Resumen OpenAI")
        path = self.write_transcription(name="meeting_transcription.txt")
        output = cli.summary_openai(path, larm_args(lang="es"))
        with open(output, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Resumen OpenAI")
        self.assertEqual(self.backend.requests[0]["model"], "gpt-4o")

    def test_gemini_provider_still_generates_a_summary(self):
        generated = {"text": "Resumen Gemini"}

        class FakeGeminiModels:
            def generate_content(self, model, contents):
                generated["model"] = model
                return types.SimpleNamespace(text="Resumen Gemini")

        class FakeGeminiClient:
            files = types.SimpleNamespace(upload=lambda file: file)
            models = FakeGeminiModels()

        google = types.ModuleType("google")
        genai = types.ModuleType("google.genai")
        genai.Client = lambda api_key=None: FakeGeminiClient()
        google.genai = genai
        path = self.write_transcription(name="meeting_transcription.txt")
        with patch.dict(sys.modules, {"google": google, "google.genai": genai}):
            output = cli.summary_gemini(path, larm_args(lang="es"))
        with open(output, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Resumen Gemini")
        self.assertEqual(generated["model"], "gemini-3.5-flash")

    def test_local_provider_still_generates_a_summary(self):
        transformers = types.ModuleType("transformers")
        transformers.pipeline = lambda task, model: (
            lambda text, max_length, min_length, do_sample: [{"summary_text": "Resumen local"}]
        )
        path = self.write_transcription(name="meeting_transcription.txt")
        with patch.dict(sys.modules, {"transformers": transformers}):
            output = cli.summary_local(path)
        with open(output, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Resumen local")

    # --------------------------------------------------------------- TEST-11
    def test_process_flow_uses_larm_and_stores_the_result(self):
        self.backend.response = build_response("Resumen del proceso")
        audio = os.path.join(self.root, "meeting_2026.mp3")
        open(audio, "wb").close()
        transcript = self.write_transcription(name="meeting_transcription_local.txt")
        args = larm_args(output=self.root, larm_model="alias")

        with patch.object(cli, "record_audio", return_value=audio), \
                patch.object(cli, "transcript_local", return_value=transcript):
            cli.action_process(args)

        rows = self.database_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][3], "larm")
        self.assertTrue(rows[0][2].endswith("_summary_larm.txt"))
        with open(rows[0][2], encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Resumen del proceso")

    def test_process_flow_keeps_previous_providers(self):
        audio = os.path.join(self.root, "meeting_2026.mp3")
        open(audio, "wb").close()
        transcript = self.write_transcription(name="meeting_transcription_local.txt")
        args = larm_args(model="openai", output=self.root, larm_model="alias")
        self.backend.response = build_response("Resumen OpenAI")

        with patch.object(cli, "record_audio", return_value=audio), \
                patch.object(cli, "transcript_openai", return_value=transcript):
            cli.action_process(args)

        rows = self.database_rows()
        self.assertEqual(rows[0][3], "openai")
        self.assertTrue(rows[0][2].endswith("summary.txt"))

    # --------------------------------------------------------------- TEST-12
    def test_interruption_does_not_lose_the_transcription(self):
        self.backend.error = KeyboardInterrupt()
        audio = os.path.join(self.root, "meeting.mp3")
        open(audio, "wb").close()
        transcript = self.write_transcription(name="meeting_transcription_local.txt")
        args = larm_args(audio=audio, output=self.root, larm_model="alias")

        with patch.object(cli, "transcript_local", return_value=transcript):
            with self.assertRaises(KeyboardInterrupt):
                cli.action_transcript(args)

        self.assertTrue(os.path.isfile(transcript))
        with open(transcript, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Contenido real.")
        rows = self.database_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], transcript)
        self.assertIsNone(rows[0][2])

    def test_interruption_returns_the_interrupt_exit_code(self):
        self.backend.error = KeyboardInterrupt()
        audio = os.path.join(self.root, "meeting.mp3")
        open(audio, "wb").close()
        transcript = self.write_transcription(name="meeting_transcription_local.txt")

        with patch.object(sys, "argv",
                          ["meeting", "transcript", "--model", "larm", "--audio", audio,
                           "--export-md", "--larm-model", "alias"]), \
                patch.object(cli, "transcript_local", return_value=transcript), \
                patch.dict(os.environ, {"LARM_MODEL": "alias"}):
            self.assertEqual(cli.main(), 130)

        self.assertTrue(os.path.isfile(transcript))
        with open(transcript, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "Contenido real.")

    # ------------------------------------------------------------------ extras
    def test_available_models_can_be_listed(self):
        self.backend.models = ["code-agent", "other-alias"]
        with larm_env(LARM_MODEL="alias"):
            config = larm.resolve_config(None)
            self.assertEqual(larm.list_models(config), ["code-agent", "other-alias"])

    def test_clean_summary_text_removes_wrapping_fences(self):
        cleaned = larm.clean_summary_text("```markdown\nResumen\n```")
        self.assertEqual(cleaned, "Resumen")


if __name__ == "__main__":
    unittest.main()

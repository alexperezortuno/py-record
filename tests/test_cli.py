import os
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

from meeting import cli


def capture_args(**overrides):
    values = {
        "monitor": "monitor.source",
        "mic": "mic.source",
        "mic_only": False,
        "system_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


class FakeProcess:
    def __init__(self, output_path):
        self.returncode = 0
        self.completed = False
        self.stdin = self
        self.output_path = output_path

    def poll(self):
        return self.returncode if self.completed else None

    def wait(self, timeout=None):
        with open(self.output_path, "wb") as output:
            output.write(b"valid mp3")
        self.completed = True
        return self.returncode

    def write(self, data):
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


class InterruptingProcess(FakeProcess):
    def __init__(self, output_path):
        super().__init__(output_path)
        self.wait_calls = 0

    def poll(self):
        return None if self.wait_calls == 0 else self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise KeyboardInterrupt
        return super().wait(timeout)


class CliTests(unittest.TestCase):
    def test_ffmpeg_commands_for_all_linux_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "meeting.mp3")
            combined = cli.build_ffmpeg_command(capture_args(), output, "linux")
            mic = cli.build_ffmpeg_command(capture_args(mic_only=True), output, "linux")
            system = cli.build_ffmpeg_command(capture_args(system_only=True), output, "linux")

        self.assertIn("amerge=inputs=2", combined)
        self.assertNotIn("amerge=inputs=2", mic)
        self.assertNotIn("amerge=inputs=2", system)
        self.assertEqual(combined[-1], output)
        self.assertEqual(mic[-1], output)
        self.assertEqual(system[-1], output)
        self.assertIn("monitor.source", system)

    def test_missing_ffmpeg_is_reported(self):
        with patch.object(cli.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
                cli.validate_audio_devices(capture_args(), "linux")

    def test_recording_process_is_waited_and_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = capture_args()
            args.output = directory
            with patch.object(cli, "validate_audio_devices"), \
                    patch.object(cli.subprocess, "Popen",
                                 side_effect=lambda command, **kwargs: FakeProcess(command[-1])):
                path = cli.record_audio(args)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_ctrl_c_finishes_recording_process(self):
        with tempfile.TemporaryDirectory() as directory:
            args = capture_args()
            args.output = directory
            with patch.object(cli, "validate_audio_devices"), \
                    patch.object(cli.subprocess, "Popen",
                                 side_effect=lambda command, **kwargs: InterruptingProcess(command[-1])):
                path = cli.record_audio(args)
            self.assertTrue(os.path.isfile(path))

    def test_cli_accepts_existing_mode_alias(self):
        with patch.object(cli, "action_transcript") as action:
            with patch.object(sys, "argv", ["meeting", "transcript", "--mode", "local", "--audio", "x.mp3"]):
                self.assertEqual(cli.main(), 0)
        self.assertEqual(action.call_args.args[0].model, "local")

    def test_cli_rejects_conflicting_capture_modes(self):
        with patch.object(sys, "argv", ["meeting", "record", "--mic-only", "--system-only"]):
            with self.assertRaises(SystemExit) as raised:
                cli.main()
        self.assertEqual(raised.exception.code, 2)

    def test_local_transcription_selects_model_device_and_timestamps(self):
        class Segment:
            start = 1.25
            end = 2.5
            text = " Hello "

        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def transcribe(self, path, beam_size):
                return iter([Segment()]), None

        module = types.ModuleType("faster_whisper")
        module.WhisperModel = FakeModel
        with tempfile.TemporaryDirectory() as directory:
            audio = os.path.join(directory, "audio.mp3")
            open(audio, "wb").close()
            with patch.dict(sys.modules, {"faster_whisper": module}):
                output = cli.transcript_local(audio, "tiny", "cpu")
            with open(output, encoding="utf-8") as transcript:
                self.assertIn("[1.25 --> 2.50] Hello", transcript.read())

    def test_cuda_requires_available_device(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            cli.resolve_whisper_device("cuda")

    def test_local_transcription_rejects_missing_audio(self):
        with self.assertRaises(FileNotFoundError):
            cli.transcript_local("missing.mp3")


if __name__ == "__main__":
    unittest.main()

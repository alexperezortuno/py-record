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
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 0
        ctranslate2.get_supported_compute_types = lambda device: set()
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*CTranslate2 detected no CUDA devices"):
                cli.resolve_whisper_device("cuda")

    def test_cpu_device_does_not_require_cuda_runtime(self):
        with patch.dict(sys.modules, {"ctranslate2": None}):
            self.assertEqual(cli.resolve_whisper_device("cpu"), "cpu")

    def test_cuda_device_uses_ctranslate2_capabilities(self):
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16", "int8_float16"}
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}), \
                patch.object(cli, "get_cuda_free_memory_mb", return_value=8192):
            self.assertEqual(cli.resolve_whisper_device("cuda"), "cuda")

    def test_auto_uses_cuda_when_ctranslate2_supports_it(self):
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}), \
                patch.object(cli, "get_cuda_free_memory_mb", return_value=8192):
            self.assertEqual(cli.resolve_whisper_device("auto"), "cuda")

    def test_auto_falls_back_to_cpu_without_cuda(self):
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 0
        ctranslate2.get_supported_compute_types = lambda device: set()
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}):
            self.assertEqual(cli.resolve_whisper_device("auto"), "cpu")

    def test_auto_uses_cpu_when_the_gpu_has_not_enough_free_memory(self):
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}), \
                patch.object(cli, "get_cuda_free_memory_mb", return_value=252):
            self.assertEqual(cli.resolve_whisper_device("auto", "small"), "cpu")

    def test_cuda_fails_when_the_gpu_has_not_enough_free_memory(self):
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}), \
                patch.object(cli, "get_cuda_free_memory_mb", return_value=252):
            with self.assertRaisesRegex(RuntimeError, "not have enough free memory"):
                cli.resolve_whisper_device("cuda", "small")

    def test_cuda_is_used_when_the_free_memory_cannot_be_measured(self):
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        with patch.dict(sys.modules, {"ctranslate2": ctranslate2}), \
                patch.object(cli, "get_cuda_free_memory_mb", return_value=None):
            self.assertEqual(cli.resolve_whisper_device("auto", "small"), "cuda")

    def test_free_memory_is_read_from_the_first_cuda_device(self):
        output = " 252\n 8192\n"

        class Result:
            returncode = 0
            stdout = output

        with patch.object(cli.subprocess, "run", return_value=Result()):
            self.assertEqual(cli.get_cuda_free_memory_mb(), 252)

    def test_free_memory_is_unknown_when_nvidia_smi_fails(self):
        class Result:
            returncode = 1
            stdout = ""

        with patch.object(cli.subprocess, "run", return_value=Result()):
            self.assertIsNone(cli.get_cuda_free_memory_mb())

    def test_free_memory_is_unknown_when_nvidia_smi_is_missing(self):
        with patch.object(cli.subprocess, "run", side_effect=FileNotFoundError()):
            self.assertIsNone(cli.get_cuda_free_memory_mb())

    def test_vram_estimate_depends_on_the_model(self):
        self.assertEqual(cli.estimate_whisper_vram_mb("tiny"), 1024)
        self.assertEqual(cli.estimate_whisper_vram_mb("small"), 2048)
        self.assertEqual(cli.estimate_whisper_vram_mb("small.en"), 2048)
        self.assertEqual(cli.estimate_whisper_vram_mb("large-v3"), 6144)
        self.assertEqual(cli.estimate_whisper_vram_mb("custom-model"), cli.DEFAULT_WHISPER_VRAM_MB)

    def test_cuda_transcription_does_not_import_torch(self):
        class Segment:
            start = 0.0
            end = 0.5
            text = " CUDA text "

        class FakeModel:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def transcribe(self, path, beam_size):
                return iter([Segment()]), None

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = FakeModel
        with tempfile.TemporaryDirectory() as directory:
            audio = os.path.join(directory, "audio.mp3")
            open(audio, "wb").close()
            with patch.dict(sys.modules, {
                "ctranslate2": ctranslate2,
                "faster_whisper": faster_whisper,
                "torch": None,
            }), patch.object(cli, "get_cuda_free_memory_mb", return_value=8192):
                output = cli.transcript_local(audio, "tiny", "cuda")
            with open(output, encoding="utf-8") as transcript:
                self.assertIn("CUDA text", transcript.read())

    def test_cuda_requires_available_device_without_ctranslate2(self):
        with patch.dict(sys.modules, {"ctranslate2": None}):
            with self.assertRaisesRegex(RuntimeError, "CUDA.*CTranslate2 is not installed"):
                cli.resolve_whisper_device("cuda")

    def test_local_transcription_rejects_missing_audio(self):
        with self.assertRaises(FileNotFoundError):
            cli.transcript_local("missing.mp3")

    def test_auto_retries_on_cpu_when_cuda_runs_out_of_memory(self):
        class Segment:
            start = 0.5
            end = 1.5
            text = " CPU text "

        devices = []

        class FakeModel:
            def __init__(self, *args, **kwargs):
                devices.append(kwargs.get("device"))
                if kwargs.get("device") == "cuda":
                    raise RuntimeError("CUDA failed with error out of memory")

            def transcribe(self, path, beam_size):
                return iter([Segment()]), None

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = FakeModel
        with tempfile.TemporaryDirectory() as directory:
            audio = os.path.join(directory, "audio.mp3")
            open(audio, "wb").close()
            with patch.dict(sys.modules, {
                "ctranslate2": ctranslate2,
                "faster_whisper": faster_whisper,
            }), patch.object(cli, "get_cuda_free_memory_mb", return_value=8192):
                output = cli.transcript_local(audio, "small", "auto")
            with open(output, encoding="utf-8") as transcript:
                self.assertIn("[0.50 --> 1.50] CPU text", transcript.read())
        self.assertEqual(devices, ["cuda", "cpu"])

    def test_explicit_cuda_does_not_fall_back_to_cpu(self):
        devices = []

        class FakeModel:
            def __init__(self, *args, **kwargs):
                devices.append(kwargs.get("device"))
                raise RuntimeError("CUDA failed with error out of memory")

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = FakeModel
        with tempfile.TemporaryDirectory() as directory:
            audio = os.path.join(directory, "audio.mp3")
            open(audio, "wb").close()
            with patch.dict(sys.modules, {
                "ctranslate2": ctranslate2,
                "faster_whisper": faster_whisper,
            }), patch.object(cli, "get_cuda_free_memory_mb", return_value=8192):
                with self.assertRaisesRegex(RuntimeError, "CUDA failed with error out of memory"):
                    cli.transcript_local(audio, "small", "cuda")
        self.assertEqual(devices, ["cuda"])

    def test_cuda_failure_without_out_of_memory_does_not_retry_on_cpu(self):
        devices = []

        class FakeModel:
            def __init__(self, *args, **kwargs):
                devices.append(kwargs.get("device"))
                raise RuntimeError("CUDA failed with error invalid device")

        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 1
        ctranslate2.get_supported_compute_types = lambda device: {"float16"}
        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = FakeModel
        with tempfile.TemporaryDirectory() as directory:
            audio = os.path.join(directory, "audio.mp3")
            open(audio, "wb").close()
            with patch.dict(sys.modules, {
                "ctranslate2": ctranslate2,
                "faster_whisper": faster_whisper,
            }), patch.object(cli, "get_cuda_free_memory_mb", return_value=8192):
                with self.assertRaisesRegex(RuntimeError, "invalid device"):
                    cli.transcript_local(audio, "small", "auto")
        self.assertEqual(devices, ["cuda"])

    def test_cuda_out_of_memory_is_detected(self):
        self.assertTrue(cli.is_cuda_out_of_memory(RuntimeError("CUDA failed with error out of memory")))
        self.assertTrue(cli.is_cuda_out_of_memory(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")))
        self.assertFalse(cli.is_cuda_out_of_memory(RuntimeError("CUDA failed with error invalid device")))


if __name__ == "__main__":
    unittest.main()

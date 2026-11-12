# Meeting CLI Assistant

A powerful command-line tool for recording, transcribing, and summarizing meetings with speaker diarization support.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Examples](#examples)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Extensibility](#extensibility)

---

## Features

| Command | Description |
|---------|-------------|
| `record` | Records audio from system output and microphone until stopped |
| `transcript` | Transcribes an audio file and generates an AI-powered summary |
| `process` | Records, transcribes, and summarizes in one continuous flow |
| `diarize` | Transcribes audio with automatic speaker identification |
| `analyze` | Extracts topics, decisions, action items and open questions from an existing transcription |

**Supported Modes (`--model`):**
- **Local** - Uses Faster-Whisper (default, completely local and no API needed)
- **OpenAI** - Uses OpenAI's Whisper API
- **Gemini** - Uses Google Gemini for intelligent summarization
- **LARM** - Transcribes locally with Faster-Whisper and summarizes with a local model served by LARM

---

## Prerequisites

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg pavucontrol cmake pkg-config libsentencepiece-dev -y
```

FFmpeg must be installed and available in `PATH`. On Linux, the recording devices
are PulseAudio/PipeWire source names; use `pactl list short sources` to inspect them.
Linux supports microphone-only, system-only, and combined capture. Windows uses
DirectShow microphone capture; Linux PulseAudio modes are not available on macOS.

**macOS:**
```bash
brew install ffmpeg cmake
```

**Python:**
- Python 3.9 or higher

---

## Installation

```bash
# Clone the repository
git clone https://github.com/alexperezortuno/py-record.git
cd py-record

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

---

## Configuration

### Audio Devices

List available audio devices on Linux:
```bash
pactl list short sources   # Input devices (microphones)
pactl list short sinks    # Output devices (speakers)
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Default audio devices
DEFAULT_MONITOR=alsa_output.pci-0000_00_1b.0.analog-stereo.monitor
DEFAULT_MIC=alsa_input.pci-0000_00_1b.0.analog-stereo

# AI Configuration
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key

# LARM (local models)
LARM_BASE_URL=http://127.0.0.1:8090/v1
LARM_MODEL=code-agent
LARM_TIMEOUT=600
LARM_API_KEY=local

# Custom prompt for summarization
SYSTEM_PROMPT=Provide a detailed summary with key action items and decisions.
```

### LARM (local models)

LARM serves local models through an OpenAI compatible API. py-record only sends the
summary request: LARM owns the model lifecycle (loading, leases and unload).

| Variable | Purpose | Default |
|----------|---------|---------|
| `LARM_BASE_URL` | Base URL of the LARM API (must include `/v1`) | `http://127.0.0.1:8090/v1` |
| `LARM_MODEL` | Alias of the model registered in LARM | none (required) |
| `LARM_TIMEOUT` | Inference timeout, in seconds | `600` |
| `LARM_API_KEY` | Value sent to the OpenAI compatible client | `local` |

`LARM_API_KEY=local` is only a compatibility value for the client: py-record does not
implement authentication. The model alias is registered per LARM installation, so list
the available aliases before using one:

```bash
curl -s http://127.0.0.1:8090/v1/models
```

Use `--larm-model` to select the alias for a single run; it overrides `LARM_MODEL`.

The same configuration serves the summary (`meeting transcript --model larm`,
`meeting process --model larm`) and the structured analysis (`meeting analyze`).
Both send the whole transcription to LARM and both need the alias to be registered
in the instance. `meeting analyze` requires a model able to answer with JSON: the
answer is validated by py-record, and the request never uses `response_format`,
JSON Schema, `temperature` or any other optional sampling parameter, so it works
with the local models served by LARM.

---

## Usage

```bash
meeting <command> [options]
```

The equivalent development invocation is `python -m meeting.cli <command> [options]`.

### Commands

#### `record`

Record audio from configured devices.

```bash
meeting record [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--monitor` | Output audio device | `DEFAULT_MONITOR` env var |
| `--mic` | Microphone device | `DEFAULT_MIC` env var |
| `--mic-only` | Capture only the microphone | `false` |
| `--system-only` | Capture only the system output monitor | `false` |
| `--output` | Recording output directory | `~/meetings` |
| `--lang` | Language code (en, es, fr, etc.) | `en` |
| `--env` | Path to `.env` file | - |

#### `transcript`

Transcribe and summarize an audio file.

```bash
meeting transcript --audio path/to/audio.mp3 [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to audio file (required) | - |
| `--model`, `--mode` | Transcription mode: `local`, `openai`, `gemini`, `larm` | `local` |
| `--prompt` | System prompt for summarization | `SYSTEM_PROMPT` env var |
| `--export-md` | Export summary as Markdown | `false` |
| `--lang` | Language code | `en` |
| `--env` | Path to `.env` file | - |
| `--whisper-model` | Faster-Whisper model name for local mode | `WHISPER_MODEL` or `small` |
| `--whisper-device` | Local device: `auto`, `cpu`, or `cuda` | `WHISPER_DEVICE` or `auto` |
| `--larm-model` | Model alias registered in LARM (only for `--model larm`) | `LARM_MODEL` env var |

With `--model larm` the audio is transcribed locally with Faster-Whisper and the
resulting transcript is summarized by the local model served by LARM. The summary is
generated together with `--export-md`, as it happens with the other providers.

#### Local device selection

CUDA availability is detected through CTranslate2 (the runtime behind Faster-Whisper),
not through PyTorch, so PyTorch is not required for GPU transcription:

- `auto` selects `cuda` when CTranslate2 reports a CUDA device with `float16` support,
  and falls back to `cpu` otherwise.
- `cpu` always uses the CPU.
- `cuda` fails with an explicit error when CTranslate2 is missing, reports no CUDA
  device, or cannot provide `float16` compute. It never silently falls back to CPU.

Before choosing the device, the free video memory of the GPU is compared with the memory
needed by the selected model (for example about 2 GB for `small` and about 6 GB for
`large-v3`). When the GPU is busy, for example because a local LLM server is running,
`auto` transcribes on CPU right away instead of failing. The check is skipped when the
free memory cannot be measured.

With `auto`, if the CUDA run fails with an out of memory error (for example when another
process, such as a local LLM server, already occupies the GPU), the transcription is
retried on CPU automatically and the run continues. An explicit `--whisper-device cuda`
keeps failing instead of falling back, and the error suggests `--whisper-device cpu`.

#### `process`

Combined recording, transcription, and summarization.

```bash
meeting process [options]
```

Same options as `transcript`, plus `--monitor`, `--mic`, `--mic-only`,
`--system-only`, and `--output` from `record`. The `process` command always generates
the summary, also with `--model larm`.

#### `diarize`

Speaker diarization with transcription.

```bash
meeting diarize --audio path/to/audio.mp3 [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to audio file (required) | - |
| `--lang` | Language code | `en` |
| `--env` | Path to `.env` file | - |

#### `analyze`

Analyze an **existing** transcription with a local model served by LARM and export
the topics, decisions, action items and open questions that the model can back with
a literal quote of the transcript.

```bash
meeting analyze --transcript path/to/transcription.txt [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--transcript`, `-t` | Path to the transcription text file (required) | - |
| `--provider` | Analysis provider: only `larm` | `larm` |
| `--larm-model` | Model alias registered in LARM (overrides `LARM_MODEL`) | `LARM_MODEL` env var |
| `--format` | Exported format: `json`, `markdown` or `both` | `both` |
| `--output` | Directory for the analysis files | the folder of the transcript |
| `--overwrite` | Replace the analysis files when they already exist | `false` |
| `--env` | Path to `.env` file | - |

The command never transcribes the audio again and never needs the original audio
file: it only reads the transcription, which is never modified. The generated files
are named after the transcript (`reunion.txt` produces `reunion_analysis.json` and
`reunion_analysis.md`).

Existing analysis files are never replaced silently: the command reports the
conflict and exits with code 1, and the files are only replaced when `--overwrite`
is given.

The answer of the model is not accepted as-is. It must be a JSON object that
follows the schema below, and every element must be backed by evidence:

- `quote` must exist literally in the transcript: only the whitespace is
  normalized, so a paraphrase is rejected. A difference limited to the letter
  case is accepted, and the fragment **of the transcript** is the one that is
  stored, never the text written by the model.
- `start_seconds` / `end_seconds` must correspond to the segment of the
  transcript that contains the quote. The model may write the time mark of the
  transcript (`04:48`) instead of a number of seconds: that reading is only
  accepted when the segment of the transcript confirms it. When the model does
  not declare the timestamps, they are anchored in the segment that holds the
  verified quote, so they always come from the transcript itself. They stay
  `null` when the transcript has no time marks.
- an element whose evidence cannot be verified is rejected and reported; no
  alternative quote is invented and the model is not asked again.

### Analysis JSON structure

```json
{
  "schema_version": "1.0",
  "topics": [
    {
      "id": "T1",
      "title": "Objetivo del sprint",
      "description": "Se repasó el objetivo del sprint.",
      "evidence": [
        {"quote": "el objetivo de este sprint", "start_seconds": 0.0, "end_seconds": 40.0}
      ]
    }
  ],
  "decisions": [
    {
      "id": "D1",
      "description": "Se decidió usar el plan B.",
      "evidence": [
        {"quote": "decidimos usar el plan B", "start_seconds": 7.1, "end_seconds": 12.0}
      ]
    }
  ],
  "action_items": [
    {
      "id": "A1",
      "description": "Actualizar la documentación.",
      "assignee": "Luis",
      "due_date": null,
      "evidence": [
        {"quote": "Luis actualiza la documentación", "start_seconds": 18.3, "end_seconds": 24.0}
      ]
    }
  ],
  "open_questions": [
    {
      "id": "Q1",
      "question": "¿Quién actualiza la documentación?",
      "evidence": [
        {"quote": "¿Quién actualiza la documentación?", "start_seconds": 12.0, "end_seconds": 18.3}
      ]
    }
  ]
}
```

`assignee` is `null` when the transcript does not name the responsible person and
`due_date` is `null` when the transcript does not state a deadline. Both are never
inferred from the context, and the language of the transcript is preserved in the
descriptions.

---

## Examples

### Basic Recording

```bash
# Record meeting in English
meeting record --lang en

# Record with specific devices
meeting record --monitor pulse --mic alsa_input --lang es

# Record only one source
meeting record --mic-only
meeting record --system-only --monitor alsa_output.example.monitor
```

### Transcription

```bash
# Local transcription (no API key needed)
meeting transcript --audio meeting.mp3

# Local transcription using a selected model and device
meeting transcript --audio meeting.mp3 --whisper-model small --whisper-device cpu

# With OpenAI Whisper API
meeting transcript --audio meeting.mp3 --mode openai

# With Gemini summarization and Markdown export
meeting transcript --audio meeting.mp3 --mode gemini --export-md

# Summarize with a local model served by LARM
meeting transcript --audio meeting.mp3 --model larm --larm-model code-agent --export-md
```

### Full Processing Pipeline

```bash
# Record and process in one command
meeting process --lang en --mode gemini --export-md

# Record, transcribe locally and summarize with LARM
meeting process --lang es --model larm --larm-model code-agent --export-md
```

### Speaker Diarization

```bash
# Identify speakers in Spanish audio
meeting diarize --audio meeting.mp3 --lang es
```

### Structured Analysis

```bash
# Analyze an existing transcription with a local model served by LARM
meeting analyze --transcript reunion.txt

# Select the alias and export only the JSON
meeting analyze --transcript reunion.txt --larm-model code-agent --format json

# Export both formats into another directory and replace previous results
meeting analyze --transcript reunion.txt --format both --output ./analysis --overwrite
```

The command produces `reunion_analysis.json` and `reunion_analysis.md` next to
the transcript (or inside `--output`).

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEFAULT_MONITOR` | Default system audio capture device | No |
| `DEFAULT_MIC` | Default microphone device | No |
| `SYSTEM_PROMPT` | Custom prompt for AI summarization | No |
| `WHISPER_MODEL` | Default Faster-Whisper model for local transcription | No |
| `WHISPER_DEVICE` | Default local device: `auto`, `cpu`, or `cuda` | No |
| `OPENAI_API_KEY` | OpenAI API key for Whisper transcription | Only for `--mode openai` |
| `GOOGLE_API_KEY` | Google API key for Gemini summarization | Only for `--mode gemini` |
| `LARM_BASE_URL` | Base URL of the LARM API (must include `/v1`) | No (`http://127.0.0.1:8090/v1`) |
| `LARM_MODEL` | Model alias registered in LARM | Only for `--model larm` |
| `LARM_TIMEOUT` | LARM inference timeout in seconds | No (`600`) |
| `LARM_API_KEY` | Value sent to the OpenAI compatible LARM client | No (`local`) |

---

## Known limitations

- `--model larm` only summarizes: the audio is always transcribed locally with
  Faster-Whisper, because LARM serves text models.
- The whole transcription is sent to LARM. A transcript larger than the context window
  of the selected model is reported as an error instead of being summarized partially:
  processing a transcript in segments is not implemented yet.
- The quality of the summary depends on the local model. Small models can add
  information that is not present in the transcript, so the summary must be reviewed.
- When LARM fails, py-record does not fall back to OpenAI or Gemini. The transcription
  is preserved (file and database row) and the error is reported with exit code 1.
- `meeting analyze` requires a UTF-8 transcription: a file with another encoding is
  reported as an error instead of being converted, so the quotes are always compared
  against the same text that the model read.
- `meeting analyze` only validates that a quote exists in the transcription and that
  its timestamps match the source segment. It does not prove that the interpretation
  of the model is correct, so the extracted elements must be reviewed.
- `meeting analyze` rejects the whole element when one of its quotes cannot be
  verified. Small local models often paraphrase a fragment or misread a time mark,
  so the number of exported elements can be much lower than the number of elements
  returned by the model; the rejected ones are reported in the log.
- `meeting analyze` sends the whole transcription to the model. A transcript larger
  than the context window of the selected model is reported as an error instead of
  being analyzed partially: processing a transcript in segments is not implemented yet.
- `meeting analyze` does not write anything in the meetings database: it neither
  records nor transcribes, it only reads an existing transcript.
- Keep `LARM_TIMEOUT` at a realistic value (the default 600 s matches the inference
  timeout of LARM). A timeout that is too short aborts the request before LARM can
  finalize it: LARM then keeps that inference as `UNKNOWN` and blocks new requests
  until the LARM instance is restarted.

---

## Dependencies

```
openai==2.3.0
faster-whisper==1.2.0
transformers==4.57.1
torch==2.8.0
sentencepiece==0.2.1
pydub==0.25.1
whisperx==3.7.2
google-genai==1.41.0
charset-normalizer==3.0.0
```

---

## Extensibility

The CLI follows a modular architecture where each subcommand maps to a dedicated function:

- `action_record()` - Audio recording
- `action_transcript()` - Transcription with summarization
- `action_process()` - Combined workflow
- `action_diarize()` - Speaker diarization
- `action_analyze()` - Structured analysis of an existing transcription

To add new commands, simply implement a new action function and register it in the CLI parser.

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

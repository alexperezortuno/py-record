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

**Supported Transcription Modes:**
- **Local** - Uses Faster-Whisper (default, completely local and no API needed)
- **OpenAI** - Uses OpenAI's Whisper API
- **Gemini** - Uses Google Gemini for intelligent summarization

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

# Custom prompt for summarization
SYSTEM_PROMPT=Provide a detailed summary with key action items and decisions.
```

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
| `--model`, `--mode` | Transcription mode: `local`, `openai`, `gemini` | `local` |
| `--prompt` | System prompt for summarization | `SYSTEM_PROMPT` env var |
| `--export-md` | Export summary as Markdown | `false` |
| `--lang` | Language code | `en` |
| `--env` | Path to `.env` file | - |
| `--whisper-model` | Faster-Whisper model name for local mode | `WHISPER_MODEL` or `small` |
| `--whisper-device` | Local device: `auto`, `cpu`, or `cuda` | `WHISPER_DEVICE` or `auto` |

#### Local device selection

CUDA availability is detected through CTranslate2 (the runtime behind Faster-Whisper),
not through PyTorch, so PyTorch is not required for GPU transcription:

- `auto` selects `cuda` when CTranslate2 reports a CUDA device with `float16` support,
  and falls back to `cpu` otherwise.
- `cpu` always uses the CPU.
- `cuda` fails with an explicit error when CTranslate2 is missing, reports no CUDA
  device, or cannot provide `float16` compute. It never silently falls back to CPU.

#### `process`

Combined recording, transcription, and summarization.

```bash
meeting process [options]
```

Same options as `transcript`, plus `--monitor`, `--mic`, `--mic-only`,
`--system-only`, and `--output` from `record`.

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
```

### Full Processing Pipeline

```bash
# Record and process in one command
meeting process --lang en --mode gemini --export-md
```

### Speaker Diarization

```bash
# Identify speakers in Spanish audio
meeting diarize --audio meeting.mp3 --lang es
```

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

To add new commands, simply implement a new action function and register it in the CLI parser.

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

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
- **Local** - Uses Faster-Whisper (default, no API needed)
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
python main.py <command> [options]
```

If installed with `pip install -e .`, replace `python main.py` with `meeting`.

### Commands

#### `record`

Record audio from configured devices.

```bash
python main.py record [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--monitor` | Output audio device | `DEFAULT_MONITOR` env var |
| `--mic` | Microphone device | `DEFAULT_MIC` env var |
| `--lang` | Language code (en, es, fr, etc.) | `en` |
| `--env` | Path to `.env` file | - |

#### `transcript`

Transcribe and summarize an audio file.

```bash
python main.py transcript --audio path/to/audio.mp3 [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to audio file (required) | - |
| `--mode` | Transcription mode: `local`, `openai`, `gemini` | `local` |
| `--prompt` | System prompt for summarization | `SYSTEM_PROMPT` env var |
| `--export-md` | Export summary as Markdown | `false` |
| `--lang` | Language code | auto-detect |
| `--env` | Path to `.env` file | - |

#### `process`

Combined recording, transcription, and summarization.

```bash
python main.py process [options]
```

Same options as `transcript`, plus device selection from `record`.

#### `diarize`

Speaker diarization with transcription.

```bash
python main.py diarize --audio path/to/audio.mp3 [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to audio file (required) | - |
| `--lang` | Language code | auto-detect |
| `--env` | Path to `.env` file | - |

---

## Examples

### Basic Recording

```bash
# Record meeting in English
python main.py record --lang en

# Record with specific devices
python main.py record --monitor pulse --mic alsa_input --lang es
```

### Transcription

```bash
# Local transcription (no API key needed)
python main.py transcript --audio meeting.mp3

# With OpenAI Whisper API
python main.py transcript --audio meeting.mp3 --mode openai

# With Gemini summarization and Markdown export
python main.py transcript --audio meeting.mp3 --mode gemini --export-md
```

### Full Processing Pipeline

```bash
# Record and process in one command
python main.py process --lang en --mode gemini --export-md
```

### Speaker Diarization

```bash
# Identify speakers in Spanish audio
python main.py diarize --audio meeting.mp3 --lang es
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEFAULT_MONITOR` | Default system audio capture device | No |
| `DEFAULT_MIC` | Default microphone device | No |
| `SYSTEM_PROMPT` | Custom prompt for AI summarization | No |
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

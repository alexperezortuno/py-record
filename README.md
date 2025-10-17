
# 🎙️ Meeting CLI Assistant

A command-line interface (CLI) tool for quickly recording, transcribing, and summarizing meetings. Ideal for users who need to capture and process meeting audio effortlessly.

---

## 🚀 Features

This CLI supports the following main operations:

- **`record`**: Records audio from output and input devices until manually stopped.
- **`transcript`**: Transcribes an `.mp3` audio file and generates a summary.
- **`process`**: Performs recording, transcription, and summarization in one continuous stream.
- **`diarize`**: Transcribes audio and identifies different speakers (diarization).

---
## 📦 Installation

```shell
# Clone the repository
git clone https://github.com/alexperezortuno/py-record.git
cd py-record

## Optional in conda or venv you need first this
pip install -U pip setuptools wheel

## Install dependencies
pip install -r requirements.txt
```

Optionally, you can create a virtual environment and install the dependencies there.

```shell
pip install -e .
```

---

## 🛠️ Requirements

### Linux

```shell
sudo apt update

sudo apt install ffmpeg pavucontrol cmake pkg-config libsentencepiece-dev -y
```

### Python 3.9+

openai==2.3.0

faster-whisper==1.2.0

transformers==4.57.1

torch==2.8.0

sentencepiece==0.2.1

pydub==0.25.1

whisperx==3.7.2

google-genai==1.41.0

charset-normalizer==3.0.0

---

### How to get audio devices

```shell
pactl list short sources
pactl list short sinks
```

---

## 🧑‍💻 Usage

Note: If you installed the package using `pip`, you can run the CLI using the `meeting` command, change the `python` command to `meeting` in the examples below.

```bash
python main.py <subcommand> [options]
```

### Available Subcommands:

---

### `record`

Record only audio using the configured devices.

```bash
python main.py record [--monitor MONITOR_DEVICE] [--mic MIC_DEVICE] [--lang LANG] [--env PATH]
```

- `--monitor`: Output audio device (default: from `DEFAULT_MONITOR` environment variable)
- `--mic`: Input microphone device (default: from `DEFAULT_MIC` environment variable)
- `--lang`: Language code (`en`, `es`, etc.)
- `--env`: Path to a `.env` file with environment variables

---

### `transcript`

Transcribe and summarize a `.mp3` audio file.

```bash
python main.py transcript --audio path/to/audio.mp3 [--mode MODE] [--prompt PROMPT] [--export-md] [--lang LANG] [--env PATH]
```

- `--audio`: Path to the `.mp3` audio file (required)
- `--mode`: `openai`, `gemini`, or `local` (default: `local`)
- `--prompt`: System prompt to generate the summary (default: from `SYSTEM_PROMPT` environment variable)
- `--export-md`: Export the summary in Markdown format
- `--lang`: Language for transcription
- `--env`: Path to the `.env` file

---

### `process`

Record, transcribe, and summarize in a single flow.

```bash
python main.py process [--mode MODE] [--prompt PROMPT] [--export-md] [--lang LANG] [--env PATH]
```

Same parameters as `transcript`, but it performs audio recording first.

---

### `diarize`

Transcribe audio and identify different speakers.

```bash
python main.py diarize --audio path/to/audio.mp3 [--lang LANG] [--env PATH]
```

- `--audio`: Path to the `.mp3` file (required)
- `--lang`: Language for transcription
- `--env`: Path to the `.env` file

---

## 🔧 Environment Variables

You can define the following variables in a `.env` file or directly in your system environment:

- `DEFAULT_MONITOR`: Default device for capturing output audio.
- `DEFAULT_MIC`: Default device for capturing microphone input.
- `SYSTEM_PROMPT`: Default system prompt used for transcription summarization.

---

## 📦 Dependencies

Ensure all required libraries are installed (e.g., `argparse`, `os`, `logging`, and the ones needed by functions like `action_record`, `action_transcript`, etc.).

You may list them in a `requirements.txt` file.

---

## 🛠️ Usage Examples

Record a meeting in English:

```bash
python main.py record --lang en
```

Transcribe a file using OpenAI and export to Markdown:

```bash
python main.py transcript --audio meeting.mp3 --mode openai --export-md
```

Process the full flow including diarization:

```bash
python main.py diarize --audio meeting.mp3 --lang es
```

---

## 📝 Notes

- The CLI automatically validates arguments and shows helpful error messages.
- You can enable verbose or debug logging in any command:
  ```bash
  python main.py process --debug
  ```

---

## 🧩 Extensibility

Each subcommand is mapped to a specific function (`action_record`, `action_transcript`, etc.), making it easy to add new functionalities without affecting the existing ones.

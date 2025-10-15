#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import signal
import sqlite3
from datetime import datetime
from openai import OpenAI
from pydub import AudioSegment
from pydub.utils import make_chunks

from meeting.diarization import Diarizer
from meeting.logger import logger_configuration

# Configuración general
DB_PATH = "meetings.db"
OUTPUT_DIR = os.path.expanduser("~/meetings")
MARKDOWN_DIR = "markdowns"
MODEL_LOCAL = "small"  # faster-whisper
MODEL_OPENAI = "gpt-4o-mini-transcribe"
MODEL_SUMMARY = "gpt-4o"  # o gpt-5 si tienes acceso

# =====================================================
# UTILIDADES
# =====================================================
def create_table():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS meetings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_date TEXT,
            transcript_name TEXT,
            audio_path TEXT,
            transcript_path TEXT,
            summary_path TEXT,
            summary_md_path TEXT,
            mode TEXT
        )
    """)
    conn.commit()
    conn.close()


def export_markdown(base_name, audio, transcription, summary):
    markdown_path = os.path.join(MARKDOWN_DIR, f"{base_name}.md")

    texto_transcripcion = open(transcription).read() if os.path.exists(transcription) else ""
    summary_text= open(summary).read() if os.path.exists(summary) else ""

    md_content = f"""# 🗓️ Meeting — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Audio:** `{audio}`  
**Transcription:** `{transcription}`  
**Resume:** `{summary}`

---

## 🧠 General summary
{text_resumen if (text_resumen := summary_text.strip()) else "Not available."}

---

## 🗣️ Full transcript
{text_transcripcion if (text_transcripcion := texto_transcripcion.strip()) else "Not available."}
"""
    with open(markdown_path, "w") as f:
        f.write(md_content)

    return markdown_path


def save_in_db(audio, transcripcion, summary, markdown_path, modo):
    base_name = os.path.basename(audio).replace(".mp3", "")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO meetings (transcript_date, transcript_name, audio_path, transcript_path, summary_path, summary_md_path, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), base_name, audio, transcripcion, summary, markdown_path, modo))
    conn.commit()
    conn.close()


def record_audio(args):
    """
    Records audio until pressed Ctrl+C.
    """
    str_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    response = os.path.join(OUTPUT_DIR, f"meeting_{str_date}.mp3")

    monitor = args.monitor
    mic = args.mic

    logger.info(f"🎙️ Recording audio in: {response}")
    logger.info("🧩 Press Ctrl+C to stop recording.\n")

    cmd = [
        "ffmpeg",
        "-f", "pulse", "-i", monitor,
        "-f", "pulse", "-i", mic,
        "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]",
        "-map", "[aout]",
        "-ac", "2", "-ar", "44100",
        response
    ]

    proceso = subprocess.Popen(cmd)

    def stop_record(sig, frame):
        logger.info("\n🛑 Recording stopped by user.")
        proceso.terminate()
        try:
            proceso.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proceso.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_record)
    proceso.wait()
    return response

def transcript_openai(audio_path):
    """
    Transcribes audio files using OpenAI's transcription API.

    This function processes an audio file and transcribes its content. If the audio
    file exceeds the duration limit supported by OpenAI, it will be split into smaller
    chunks for sequential transcription. The transcription result is saved as a text
    file with the same base name as the input audio file.

    Parameters:
    audio_path (str): The path to the audio file to be transcribed.

    Returns:
    str: The path to the generated transcription text file, or None if the input
         file does not exist.
    """
    if not os.path.exists(audio_path):
        return None
    if os.getenv("OPENAI_API_KEY") is None:
        client = OpenAI()
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("🧠 Transcribing with OpenAI...")

    # Chunk the audio if it exceeds the limit
    audio = AudioSegment.from_file(audio_path)
    audio_length_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    max_duration = 1350  # Maximum duration in seconds per OpenAI API limit

    chunks = []
    if audio_length_seconds > max_duration:
        logger.info(f"⚠️ Audio file exceeds duration limit ({max_duration}s), splitting into parts...")
        chunks = make_chunks(audio, max_duration * 1000)  # Convert seconds to milliseconds
    else:
        chunks = [audio]

    transcriptions = []
    for i, chunk in enumerate(chunks):
        tmp_chunk_path = f"{audio_path}_chunk{i}.mp3"
        chunk.export(tmp_chunk_path, format="mp3")  # Export chunk to disk
        logger.info(f"⬆️ Transcribing segment {i + 1}/{len(chunks)}...")
        with open(tmp_chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(model=MODEL_OPENAI, file=f)
            transcriptions.append(result.text)

    response = audio_path.replace(".mp3", "_transcription.txt")
    with open(response, "w") as out:
        out.write("\n".join(transcriptions))
    return response


def transcript_local(audio_path):
    """
    Transcribes an audio file into a text file using a local Whisper model.

    The function uses the WhisperModel to transcribe the given audio file and outputs
    the transcription to a text file in the same directory as the audio file. It processes
    the transcription by segments and writes each segment's text to the output file.

    Parameters:
        audio_path (str): The path to the audio file to be transcribed.

    Returns:
        str: The path to the output transcription text file.
    """
    from faster_whisper import WhisperModel
    logger.info("🧠 Transcribing locally...")
    modelo = WhisperModel(MODEL_LOCAL, device="cpu")
    segments, info = modelo.transcribe(audio_path, beam_size=5)
    salida = audio_path.replace(".mp3", "_transcription_local.txt")
    with open(salida, "w") as out:
        for seg in segments:
            out.write(seg.text.strip() + "\n")
    return salida


def summary_openai(transcription_path, args):
    """
    Generates a summarized output based on the contents of a transcription file
    using OpenAI's language model. The summary includes a general overview,
    key points, decisions or tasks along with responsibilities, and pending
    issues or next steps. The result is saved to a new file with a modified name.

    Parameters:
        transcription_path (str): Path to the transcription text file.
        args: Arguments object containing necessary configuration such as the
              prompt for the OpenAI model.

    Returns:
        str: File path of the summary output.

    Raises:
        FileNotFoundError: If the transcription file does not exist.
        OpenAIError: If OpenAI services fail during the request or response.

    """
    texto = open(transcription_path).read()
    client = OpenAI()
    logger.info("🧩 Generating summary with OpenAI...")
    prompt = f"""
    Analyze this transcript and deliver: 
    1. A general summary (max 5 paragraphs) 
    2. List of key points 
    3. Decisions or tasks with those responsible 
    4. Pending issues or next steps 

    Transcription:
    {texto[:15000]}
    """
    completion = client.chat.completions.create(
        model=MODEL_SUMMARY,
        messages=[
            {"role": "system", "content": args.prompt},
            {"role": "user", "content": prompt}
        ]
    )
    resumen = completion.choices[0].message.content
    salida = transcription_path.replace(".txt", "_resumen.txt")
    with open(salida, "w") as f:
        f.write(resumen)
    return salida


def summary_local(transcription_path):
    """
    Generates a summarized version of the text from a transcription file using a pre-trained summarization
    model.

    Parameters:
    transcription_path: str
        The file path to the transcription text file that needs to be summarized.

    Returns:
    str
        The file path of the newly created file containing the summarized text.
    """
    from transformers import pipeline
    logger.info("🧩 Generating summary locally..")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text = open(transcription_path).read()
    fragments = [text[i:i+3000] for i in range(0, len(text), 3000)]
    summaries = []
    for frag in fragments:
        summary = summarizer(frag, max_length=300, min_length=80, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    salida = transcription_path.replace(".txt", "_summary_local.txt")
    with open(salida, "w") as f:
        f.write("\n\n".join(summaries))
    return salida


def action_record(args):
    """
    Records an audio session and provides the path where the recording is saved.
    The function creates necessary tables, records an audio session based on provided arguments, and outputs the
    path of the saved recording. Additionally, it suggests a command to transcribe the recorded audio.

    Args:
        args: Arguments required for recording the audio.

    Returns:
        None
    """
    create_table()
    audio_path = record_audio(args)
    logger.info(f"\n✅ Recording saved in {audio_path}")
    logger.info(f"You can later transcribe with:\n   meeting transcript --mode local --audio {audio_path}\n")


def action_transcript(args):
    create_table()
    if not args.audio:
        logger.error("❌ You must specify a file with --audio.")
        sys.exit(1)
    audio_path = args.audio
    transcript_path = transcript_openai(audio_path) if args.mode == "online" else transcript_local(audio_path)
    summary_path = summary_openai(transcript_path, args) if args.mode == "online" else summary_local(transcript_path)
    base_name = os.path.basename(audio_path).replace(".mp3", "")
    markdown_path = ""
    if args.export_md:
        markdown_path = export_markdown(base_name, audio_path, transcript_path, summary_path)
    save_in_db(audio_path, transcript_path, summary_path, markdown_path, args.mode)
    logger.info(f"\n✅ Full transcript:\n📄 {markdown_path}\n💾 DB: {DB_PATH}\n")

def action_process(args):
    create_table()
    audio_path = record_audio(args)
    transcript_path = transcript_openai(audio_path) if args.mode == "online" else transcript_local(audio_path)
    summary_path = summary_openai(transcript_path, args) if args.mode == "online" else summary_local(transcript_path)
    base_name = os.path.basename(audio_path).replace(".mp3", "")
    markdown_path = ""
    if args.export_md:
        markdown_path = export_markdown(base_name, audio_path, transcript_path, summary_path)
    save_in_db(audio_path, transcript_path, summary_path, markdown_path, args.mode)
    logger.info(f"\n✅ Everything ready:\n📄 Markdown: {markdown_path}\n💾 DB: {DB_PATH}\n")

def action_diarize(args):
    diarizer = Diarizer()
    diarizer.transcribe_with_diarization(args.audio, lang=args.lang)


def main():
    """
    Main entry point for the meeting CLI application.

    This function initializes the command-line interface (CLI) for the meeting
    application. It sets up argument parsing, manages subcommands, and runs the
    appropriate function based on user input. The application supports three
    primary operations: 'record' for audio recording, 'transcript' for audio
    transcription and summarization, and 'process' for combining recording and
    transcription into a unified workflow. Each subcommand is configured with its
    respective arguments.

    Subcommands:
    - record: Records only audio until manually interrupted.
    - transcript: Transcribes and summarizes a recorded audio file.
    - process: Performs recording, transcription, and summarization in a single
      operation.

    Environment variables:
    - DEFAULT_MONITOR: Default device for recording audio output.
    - DEFAULT_MIC: Default device for listening to audio input.
    - SYSTEM_PROMPT: Default system prompt for transcription summarization.

    :raises SystemExit: If invalid arguments are provided or an unexpected error
        occurs during execution.
    """
    default_monitor: str = os.getenv("DEFAULT_MONITOR", "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor")
    default_mic: str = os.getenv("DEFAULT_MIC", "alsa_input.usb-Logitech_MIC-00.mono-fallback")
    default_system_prompt: str = os.getenv("SYSTEM_PROMPT", "You are an expert meeting summary assistant.")
    parser = argparse.ArgumentParser(prog="meeting", description="CLI wizard to record and transcribe meetings.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    p1 = subparsers.add_parser("record", help="Record only the audio up to Ctrl+C.")
    p1.add_argument("--monitor", default=default_monitor, help="Record device for audio output")
    p1.add_argument("--mic", default=default_mic, help="Listen device for audio input")
    p1.set_defaults(func=action_record)

    p2 = subparsers.add_parser("transcript", help="Transcribe and summarize recorded audio.")
    p2.add_argument("-m", "--mode", choices=["online", "local"], default="local", help="Transcription mode")
    p2.add_argument("-a", "--audio", required=True, help="Audio file path .mp3")
    p2.add_argument("-e", "--export-md", action='store_true', help='Export the markdown in the folder')
    p2.add_argument("-p", "--prompt", default=default_system_prompt, help="System prompt")
    p2.set_defaults(func=action_transcript)

    p3 = subparsers.add_parser("process", help="Record, transcribe and summarize in a single stream.")
    p3.add_argument("-m", "--mode", choices=["online", "local"], default="local", help="Processing mode")
    p3.add_argument("-e", "--export-md", action='store_true', help='Export the markdown in the folder')
    p3.add_argument("-p", "--prompt", default=default_system_prompt, help="System prompt")
    p3.set_defaults(func=action_process)

    p4 = subparsers.add_parser("diarize", help="Transcribe and identify speakers (diarization).")
    p4.add_argument("-a", "--audio", required=True, help="Path to the file .mp3")
    p4.add_argument("-l", "--lang", default="es", help="Language code (e.g., es, en)")
    p4.set_defaults(func=action_diarize)

    args = parser.parse_args()
    global logger
    logger = logger_configuration(verbose=args.verbose, debug=args.debug)
    args.func(args)


if __name__ == "__main__":
    main()

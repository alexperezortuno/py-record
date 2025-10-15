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

# Configuración general
DB_PATH = "reuniones.db"
OUTPUT_DIR = os.path.expanduser("~/reuniones")
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
        CREATE TABLE IF NOT EXISTS reuniones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            nombre TEXT,
            audio_path TEXT,
            transcripcion_path TEXT,
            resumen_path TEXT,
            resumen_md_path TEXT,
            modo TEXT
        )
    """)
    conn.commit()
    conn.close()


def export_markdown(nombre_base, audio, transcripcion, resumen):
    markdown_path = os.path.join(MARKDOWN_DIR, f"{nombre_base}.md")

    texto_transcripcion = open(transcripcion).read() if os.path.exists(transcripcion) else ""
    texto_resumen = open(resumen).read() if os.path.exists(resumen) else ""

    md_content = f"""# 🗓️ Reunión — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Audio:** `{audio}`  
**Transcripción:** `{transcripcion}`  
**Resumen:** `{resumen}`

---

## 🧠 Resumen general
{text_resumen if (text_resumen := texto_resumen.strip()) else "No disponible."}

---

## 🗣️ Transcripción completa
{text_transcripcion if (text_transcripcion := texto_transcripcion.strip()) else "No disponible."}
"""
    with open(markdown_path, "w") as f:
        f.write(md_content)

    return markdown_path


def registrar_en_db(audio, transcripcion, resumen, markdown_path, modo):
    nombre_base = os.path.basename(audio).replace(".mp3", "")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO reuniones (fecha, nombre, audio_path, transcripcion_path, resumen_path, resumen_md_path, modo)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), nombre_base, audio, transcripcion, resumen, markdown_path, modo))
    conn.commit()
    conn.close()

# =====================================================
# ETAPA 1 — GRABAR
# =====================================================
def record_audio(args):
    """
    Graba el audio hasta que se presione Ctrl+C.
    """
    fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    salida = os.path.join(OUTPUT_DIR, f"reunion_{fecha}.mp3")

    monitor = args.monitor
    mic = args.mic

    print(f"🎙️ Grabando audio en: {salida}")
    print("🧩 Presiona Ctrl+C para detener la grabación.\n")

    cmd = [
        "ffmpeg",
        "-f", "pulse", "-i", monitor,
        "-f", "pulse", "-i", mic,
        "-filter_complex", "[0:a][1:a]amerge=inputs=2[aout]",
        "-map", "[aout]",
        "-ac", "2", "-ar", "44100",
        salida
    ]

    proceso = subprocess.Popen(cmd)

    def stop_record(sig, frame):
        print("\n🛑 Grabación detenida por el usuario.")
        proceso.terminate()
        try:
            proceso.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proceso.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_record)
    proceso.wait()
    return salida

# =====================================================
# ETAPA 2 — TRANSCRIPCIÓN
# =====================================================
def transcript_openai(audio_path):
    if not os.path.exists(audio_path):
        return None
    if os.getenv("OPENAI_API_KEY") is None:
        client = OpenAI()
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("🧠 Transcribiendo con OpenAI...")

    # Chunk the audio if it exceeds the limit
    audio = AudioSegment.from_file(audio_path)
    audio_length_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    max_duration = 1400  # Maximum duration in seconds per OpenAI API limit

    chunks = []
    if audio_length_seconds > max_duration:
        print(f"⚠️ El archivo de audio excede el límite de duración ({max_duration}s), dividiendo en partes...")
        chunks = make_chunks(audio, max_duration * 1000)  # Convert seconds to milliseconds
    else:
        chunks = [audio]

    transcriptions = []
    for i, chunk in enumerate(chunks):
        tmp_chunk_path = f"{audio_path}_chunk{i}.mp3"
        chunk.export(tmp_chunk_path, format="mp3")  # Export chunk to disk
        print(f"⬆️ Transcribiendo segmento {i + 1}/{len(chunks)}...")
        with open(tmp_chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(model=MODEL_OPENAI, file=f)
            transcriptions.append(result.text)

    salida = audio_path.replace(".mp3", "_transcripcion.txt")
    with open(salida, "w") as out:
        out.write("\n".join(transcriptions))
    return salida


def transcript_local(audio_path):
    from faster_whisper import WhisperModel
    print("🧠 Transcribiendo localmente...")
    modelo = WhisperModel(MODEL_LOCAL, device="cpu")
    segments, info = modelo.transcribe(audio_path, beam_size=5)
    salida = audio_path.replace(".mp3", "_transcripcion_local.txt")
    with open(salida, "w") as out:
        for seg in segments:
            out.write(seg.text.strip() + "\n")
    return salida

# =====================================================
# ETAPA 3 — RESUMEN
# =====================================================
def resume_openai(transcription_path):
    texto = open(transcription_path).read()
    client = OpenAI()
    print("🧩 Generando resumen con OpenAI...")
    prompt = f"""
    Analiza esta transcripción y entrega:
    1. Un resumen general (máx 5 párrafos)
    2. Lista de puntos clave
    3. Decisiones o tareas con responsables
    4. Temas pendientes o próximos pasos

    Transcripción:
    {texto[:15000]}
    """
    completion = client.chat.completions.create(
        model=MODEL_SUMMARY,
        messages=[
            {"role": "system", "content": "Eres un asistente experto en resumen de reuniones."},
            {"role": "user", "content": prompt}
        ]
    )
    resumen = completion.choices[0].message.content
    salida = transcription_path.replace(".txt", "_resumen.txt")
    with open(salida, "w") as f:
        f.write(resumen)
    return salida


def resume_local(transcription_path):
    from transformers import pipeline
    print("🧩 Generando resumen local...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    texto = open(transcription_path).read()
    fragmentos = [texto[i:i+3000] for i in range(0, len(texto), 3000)]
    resumenes = []
    for frag in fragmentos:
        resumen = summarizer(frag, max_length=300, min_length=80, do_sample=False)
        resumenes.append(resumen[0]['summary_text'])
    salida = transcription_path.replace(".txt", "_resume_local.txt")
    with open(salida, "w") as f:
        f.write("\n\n".join(resumenes))
    return salida


def action_record(args):
    create_table()
    audio_path = record_audio(args)
    print(f"\n✅ Grabación guardada en {audio_path}")
    print(f"Puedes transcribir luego con:\n   reunion transcript --mode local --audio {audio_path}\n")

def action_transcript(args):
    create_table()
    if not args.audio:
        print("❌ Debes especificar un archivo con --audio.")
        sys.exit(1)
    audio_path = args.audio
    transcripcion_path = transcript_openai(audio_path) if args.mode == "online" else transcript_local(audio_path)
    resumen_path = resume_openai(transcripcion_path) if args.mode == "online" else resume_local(transcripcion_path)
    nombre_base = os.path.basename(audio_path).replace(".mp3", "")
    markdown_path = export_markdown(nombre_base, audio_path, transcripcion_path, resumen_path)
    registrar_en_db(audio_path, transcripcion_path, resumen_path, markdown_path, args.mode)
    print(f"\n✅ Transcripción completa:\n📄 {markdown_path}\n💾 DB: {DB_PATH}\n")

def action_process(args):
    create_table()
    audio_path = record_audio(args)
    transcripcion_path = transcript_openai(audio_path) if args.mode == "online" else transcript_local(audio_path)
    resumen_path = resume_openai(transcripcion_path) if args.mode == "online" else resume_local(transcripcion_path)
    nombre_base = os.path.basename(audio_path).replace(".mp3", "")
    markdown_path = export_markdown(nombre_base, audio_path, transcripcion_path, resumen_path)
    registrar_en_db(audio_path, transcripcion_path, resumen_path, markdown_path, args.mode)
    print(f"\n✅ Todo listo:\n📄 Markdown: {markdown_path}\n💾 DB: {DB_PATH}\n")

# =====================================================
# CLI PRINCIPAL
# =====================================================
def main():
    default_monitor: str = os.getenv("DEFAULT_MONITOR", "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor")
    default_mic: str = os.getenv("DEFAULT_MIC", "alsa_input.usb-Logitech_MIC-00.mono-fallback")
    parser = argparse.ArgumentParser(prog="reunion", description="Asistente CLI para grabar y transcribir reuniones.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    p1 = subparsers.add_parser("record", help="Graba solo el audio hasta Ctrl+C.")
    p1.add_argument("--monitor", default=default_monitor, help="Record device for audio output")
    p1.add_argument("--mic", default=default_mic, help="Listen device for audio input")
    p1.set_defaults(func=action_record)

    p2 = subparsers.add_parser("transcript", help="Transcribe y resume un audio grabado.")
    p2.add_argument("--mode", choices=["online", "local"], default="local", help="Modo de transcripción")
    p2.add_argument("--audio", required=True, help="Ruta del archivo de audio .mp3")
    p2.set_defaults(func=action_transcript)

    p3 = subparsers.add_parser("process", help="Graba, transcribe y resume en un solo flujo.")
    p3.add_argument("--mode", choices=["online", "local"], default="local", help="Modo de procesamiento")
    p3.set_defaults(func=action_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

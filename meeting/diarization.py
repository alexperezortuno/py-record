import os
from typing import Optional

import torch
import whisperx

class Diarizer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def load_diarizer(self, device="cuda" if torch.cuda.is_available() else "cpu", hf_token=None):
        """
        Devuelve una instancia de la diarización de WhisperX compatible con múltiples versiones.
        - Intenta importar desde whisperx.diarize.DiarizationPipeline (>=3.3 aprox)
        - Si no existe, intenta usar whisperx.load_diarize_model (algunas releases)
        """
        # 1) Intento moderno: submódulo diarize
        try:
            from whisperx.diarize import DiarizationPipeline
            return DiarizationPipeline(use_auth_token=hf_token, device=device)
        except Exception:
            pass

        # 2) Alternativa: loader utilitario si existe en tu build
        try:
            # Algunas versiones exponen un "loader" en el paquete
            return whisperx.load_diarize_model(device=device, use_auth_token=hf_token)
        except Exception as e:
            raise RuntimeError(
                "No pude crear el diarizer. Revisa tu versión de whisperx o usa la rama oficial más reciente.\n"
                f"Error original: {e}"
            )

    def transcribe_with_diarization(self, audio_path: str, lang: Optional[str]="es", hf_token: Optional[str]=None):
        """
        Transcribes an audio file with speaker diarization using WhisperX.

        The method processes the given audio file to determine the text transcription
        and assigns speaker labels to the dialog segments using a diarization model.
        The transcription and speaker tags are saved into a new text file.

        Parameters:
        audio_path: str
            Path to the audio file to be transcribed.
        lang: str, optional
            Language code for transcription (default is "es").
        hf_token: Optional[str], optional
            Hugging Face authentication token for model access. Required if the diarization
            model needs authentication.

        Returns:
        str
            Path to the output text file containing the transcription with speaker labels.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎙️ Processing {audio_path} with WhisperX in {device}...")

        # Ensure the hardware supports float16, fall back to float32 otherwise
        compute_type = "float16" if device == "cuda" and torch.cuda.is_available() and \
                                    torch.cuda.get_device_capability()[0] >= 7 else "float32"

        model = whisperx.load_model("large-v2", self.device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16, language=lang)

        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

        diarize_model = self.load_diarizer(device=device, hf_token=hf_token)
        diarize_segments = diarize_model(audio)

        result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)

        final_text = []
        for seg in result_aligned["segments"]:
            speaker = seg.get("speaker", "SPEAKER_XX")
            final_text.append(f"[{speaker}] {seg['text'].strip()}")

        output_txt = os.path.splitext(audio_path)[0] + "_diarizado.txt"
        with open(output_txt, "w") as f:
            f.write("\n".join(final_text))

        print(f"✅ Diarization completed: {output_txt}")
        return output_txt

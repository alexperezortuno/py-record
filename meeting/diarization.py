import os
import torch
import whisperx

class Diarizer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def transcribe_with_diarization(self, audio_path, lang="es"):
        print(f"🎙️ Processing {audio_path} with WhisperX in {self.device}...")

        model = whisperx.load_model("large-v2", self.device)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16, language=lang)

        model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, self.device)

        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=self.device)
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

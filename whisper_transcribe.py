import os
import torch
import logging
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faster_whisper")

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace(".mp3", ".wav")
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        logger.info(f"Converted MP3 to WAV: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Failed to convert MP3 to WAV: {e}")
        return None

def transcribe_audio(file_path):
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    if file_path.lower().endswith(".mp3"):
        logger.info("Converting MP3 to WAV...")
        file_path = convert_mp3_to_wav(file_path)
        if not file_path:
            return

    # Initialize Whisper model
    logger.info("Initializing Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel("medium", device=device, compute_type=compute_type)

    logger.info(f"Transcribing: {file_path}...")
    segments, info = model.transcribe(
        file_path,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.3,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 1000,
            "speech_pad_ms": 400
        },
        language="en"  # Specify language if known
    )

    logger.info(f"Detected language: {info.language} (Confidence: {info.language_probability:.2f})")

    print("\n[TRANSCRIBED TEXT]")
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == "__main__":
    file_path = input("Enter the path to your audio file: ").strip()
    transcribe_audio(file_path)

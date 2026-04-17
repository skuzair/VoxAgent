from typing import Optional

from . import get_groq_client

SUPPORTED_AUDIO_FORMATS = {
    "flac",
    "m4a",
    "mp3",
    "mp4",
    "mpeg",
    "mpga",
    "ogg",
    "wav",
    "webm",
}


class STTError(Exception):
    """Raised when speech-to-text processing fails."""


def _normalize_file_format(file_format: Optional[str]) -> str:
    normalized = (file_format or "").lower().strip().replace(".", "")
    if normalized == "oga":
        normalized = "ogg"
    if normalized == "wave":
        normalized = "wav"
    return normalized


def transcribe_audio(audio_bytes: bytes, file_format: str) -> str:
    if not audio_bytes:
        raise STTError("No audio was provided.")
    if len(audio_bytes) < 256:
        raise STTError("The audio input is too short or empty.")

    normalized_format = _normalize_file_format(file_format)
    if normalized_format not in SUPPORTED_AUDIO_FORMATS:
        raise STTError(
            f"Unsupported audio format '{file_format}'. Use wav, mp3, m4a, ogg, or webm."
        )

    try:
        transcription = get_groq_client().audio.transcriptions.create(
            file=(f"input.{normalized_format}", audio_bytes),
            model="whisper-large-v3",
            response_format="json",
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        raise STTError(
            "Could not transcribe the audio. Try again with clearer audio or check the API key."
        ) from exc

    transcript = (getattr(transcription, "text", "") or "").strip()
    if not transcript:
        raise STTError("No speech was detected in the audio.")
    if not any(char.isalnum() for char in transcript):
        raise STTError("The transcription was not intelligible enough to use.")

    return transcript

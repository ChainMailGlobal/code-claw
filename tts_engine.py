import base64
import os
import re
import subprocess
import tempfile

from config import Config


def synthesize(text: str) -> str:
    """Synthesize text to speech. Returns base64-encoded WAV string."""
    cleaned = _prepare_text(text)
    if Config.TTS_ENGINE == "coqui":
        wav_bytes = _synthesize_coqui(cleaned)
    else:
        wav_bytes = _synthesize_piper(cleaned)
    return base64.b64encode(wav_bytes).decode("utf-8")


def _prepare_text(text: str) -> str:
    """Strip code blocks, truncate to ~150 words, keep prose only."""
    # Remove markdown code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    # Remove markdown symbols
    text = re.sub(r"[#*_>\-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Truncate to 150 words
    words = text.split()
    if len(words) > 150:
        text = " ".join(words[:150]) + "."
    return text


def _synthesize_piper(text: str) -> bytes:
    model_path = Config.piper_model_path()
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Piper model not found at {model_path}. Run start.sh first."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["piper", "--model", model_path, "--output_file", tmp_path],
            input=text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed: {proc.stderr}")
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _synthesize_coqui(text: str) -> bytes:
    try:
        from TTS.api import TTS as CoquiTTS  # lazy import — heavy
    except ImportError:
        raise RuntimeError("Coqui TTS not installed. pip install TTS")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        tts = CoquiTTS(model_name="tts_models/en/ljspeech/fast_pitch")
        tts.tts_to_file(text=text, file_path=tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

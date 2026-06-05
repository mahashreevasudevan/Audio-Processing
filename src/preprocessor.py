import subprocess
import soundfile as sf
import logging
from pathlib import Path
from configs.config import DATA_RAW, DATA_PROCESSED, SAMPLE_RATE, CHANNELS

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    pass


def convert_to_wav(video_id: str, raw_ext: str) -> Path:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    input_path = DATA_RAW / f"{video_id}.{raw_ext}"
    output_path = DATA_PROCESSED / f"{video_id}_16k.wav"

    if not input_path.exists():
        raise PreprocessingError(f"Input file not found: {input_path}")

    if output_path.exists():
        logger.info(f"{video_id}: processed file already exists, skipping conversion")
        return output_path

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-acodec", "pcm_s16le",
        "-y",
        str(output_path)
    ]

    logger.info(f"{video_id}: running FFmpeg conversion")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise PreprocessingError(f"FFmpeg failed: {result.stderr}")

    return output_path


def validate_audio(output_path: Path) -> dict:
    if not output_path.exists():
        raise PreprocessingError(f"File does not exist: {output_path}")

    info = sf.info(output_path)

    if info.samplerate != SAMPLE_RATE:
        raise PreprocessingError(f"Wrong sample rate: {info.samplerate}, expected {SAMPLE_RATE}")

    if info.channels != CHANNELS:
        raise PreprocessingError(f"Wrong channel count: {info.channels}, expected {CHANNELS}")

    if info.duration < 1.0:
        raise PreprocessingError(f"Audio too short: {info.duration:.2f}s")

    if info.duration > 7200:
        logger.warning(f"Very long audio: {info.duration:.0f}s — diarization may be slow")

    logger.info(f"Validation passed: {info.duration:.1f}s, {info.samplerate}Hz, {info.channels}ch")

    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "format": info.format,
    }


def preprocess(video_id: str, raw_ext: str) -> dict:
    output_path = convert_to_wav(video_id, raw_ext)
    audio_meta = validate_audio(output_path)

    return {
        "video_id": video_id,
        "processed_path": str(output_path),
        "audio": audio_meta,
    }
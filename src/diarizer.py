import json
import logging
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from configs.config import DATA_PROCESSED, RESULTS_DIR, HF_TOKEN

logger = logging.getLogger(__name__)


class DiarizationError(Exception):
    pass


def load_pipeline() -> Pipeline:
    import os
    from huggingface_hub import login
    login(token=HF_TOKEN)
    
    logger.info("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    logger.info(f"Pipeline loaded on {device}")
    return pipeline


def run_diarization(video_id: str, pipeline: Pipeline) -> list:
    wav_path = DATA_PROCESSED / f"{video_id}_16k.wav"

    if not wav_path.exists():
        raise DiarizationError(f"Processed audio not found: {wav_path}")

    logger.info(f"{video_id}: running diarization")
    diarization = pipeline(str(wav_path))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        seg = {
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
            "duration": round(turn.end - turn.start, 3),
        }
        segments.append(seg)

    if len(segments) == 0:
        raise DiarizationError(f"{video_id}: diarization returned no segments")

    logger.info(f"{video_id}: found {len(segments)} segments, "
                f"{len(set(s['speaker'] for s in segments))} speakers")

    return segments


def save_rttm(video_id: str, segments: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rttm_path = RESULTS_DIR / f"{video_id}.rttm"

    with open(rttm_path, "w") as f:
        for seg in segments:
            duration = seg["end"] - seg["start"]
            line = (f"SPEAKER {video_id} 1 {seg['start']:.3f} {duration:.3f} "
                    f"<NA> <NA> {seg['speaker']} <NA> <NA>\n")
            f.write(line)

    logger.info(f"RTTM saved: {rttm_path}")


def validate_diarization(segments: list, audio_duration: float) -> dict:
    speakers = set(s["speaker"] for s in segments)
    total_speech = sum(s["duration"] for s in segments)
    short_segments = [s for s in segments if s["duration"] < 0.5]

    issues = []
    if len(speakers) == 1:
        issues.append("Only one speaker detected")
    if len(speakers) > 20:
        issues.append(f"Unusually high speaker count: {len(speakers)}")
    if total_speech / audio_duration < 0.1:
        issues.append("Less than 10% of audio has speech")
    if len(short_segments) / len(segments) > 0.5:
        issues.append(f"{len(short_segments)} segments under 0.5s")

    report = {
        "num_segments": len(segments),
        "num_speakers": len(speakers),
        "speakers": list(speakers),
        "total_speech_duration": round(total_speech, 2),
        "speech_ratio": round(total_speech / audio_duration, 3),
        "short_segments": len(short_segments),
        "issues": issues,
        "passed": len(issues) == 0,
    }

    if issues:
        logger.warning(f"Diarization validation issues: {issues}")
    else:
        logger.info("Diarization validation passed")

    return report


def diarize(video_id: str, pipeline: Pipeline, audio_duration: float) -> dict:
    segments = run_diarization(video_id, pipeline)
    validation = validate_diarization(segments, audio_duration)

    diar_output = {
        "video_id": video_id,
        "segments": segments,
        "validation": validation,
    }

    output_path = RESULTS_DIR / f"{video_id}_diarization.json"
    with open(output_path, "w") as f:
        json.dump(diar_output, f, indent=2)

    save_rttm(video_id, segments)

    return diar_output
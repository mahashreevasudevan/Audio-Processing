import json
import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
from transformers import pipeline as hf_pipeline
from configs.config import DATA_PROCESSED, RESULTS_DIR, MIN_SEGMENT_DURATION, EMOTION_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class EmotionTaggingError(Exception):
    pass


def load_emotion_model():
    logger.info("Loading emotion recognition model...")
    model = hf_pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device=0 if torch.cuda.is_available() else -1,
    )
    logger.info("Emotion model loaded")
    return model


def extract_segment_audio(wav_path: Path, start: float, end: float,
                           sample_rate: int = 16000) -> np.ndarray:
    data, sr = sf.read(str(wav_path))
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    return data[start_frame:end_frame]


def tag_segment(audio_array: np.ndarray, model, sample_rate: int = 16000) -> dict:
    result = model({"array": audio_array, "sampling_rate": sample_rate})
    top = result[0]
    return {
        "emotion": top["label"],
        "emotion_confidence": round(top["score"], 4),
        "all_scores": {r["label"]: round(r["score"], 4) for r in result},
    }


def tag_all_segments(video_id: str, segments: list, model) -> list:
    wav_path = DATA_PROCESSED / f"{video_id}_16k.wav"

    if not wav_path.exists():
        raise EmotionTaggingError(f"Processed audio not found: {wav_path}")

    tagged = []
    skipped = 0

    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]

        if duration < MIN_SEGMENT_DURATION:
            skipped += 1
            tagged.append({**seg, "emotion": None, "emotion_confidence": None,
                           "emotion_flag": "SKIPPED_TOO_SHORT", "all_scores": None})
            continue

        try:
            audio = extract_segment_audio(wav_path, seg["start"], seg["end"])
            emotion_result = tag_segment(audio, model)

            flag = None
            if emotion_result["emotion_confidence"] < EMOTION_CONFIDENCE_THRESHOLD:
                flag = "LOW_CONFIDENCE"
                logger.warning(f"Segment {i}: low confidence {emotion_result['emotion_confidence']:.2f} "
                               f"for emotion '{emotion_result['emotion']}'")

            tagged.append({
                **seg,
                "emotion": emotion_result["emotion"],
                "emotion_confidence": emotion_result["emotion_confidence"],
                "all_scores": emotion_result["all_scores"],
                "emotion_flag": flag,
            })

        except Exception as e:
            logger.error(f"Segment {i} emotion tagging failed: {e}")
            tagged.append({**seg, "emotion": None, "emotion_confidence": None,
                           "emotion_flag": "MODEL_ERROR", "all_scores": None})

    logger.info(f"{video_id}: tagged {len(tagged) - skipped} segments, "
                f"skipped {skipped} short segments")
    return tagged


def tag_emotions(video_id: str, diarization_output: dict, model) -> dict:
    segments = diarization_output["segments"]
    tagged_segments = tag_all_segments(video_id, segments, model)

    output = {
        "video_id": video_id,
        "tagged_segments": tagged_segments,
        "stats": {
            "total": len(tagged_segments),
            "tagged": sum(1 for s in tagged_segments if s["emotion"] is not None),
            "skipped_short": sum(1 for s in tagged_segments if s.get("emotion_flag") == "SKIPPED_TOO_SHORT"),
            "low_confidence": sum(1 for s in tagged_segments if s.get("emotion_flag") == "LOW_CONFIDENCE"),
            "model_errors": sum(1 for s in tagged_segments if s.get("emotion_flag") == "MODEL_ERROR"),
        }
    }

    output_path = RESULTS_DIR / f"{video_id}_emotion.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Emotion tagging complete: {output['stats']}")
    return output
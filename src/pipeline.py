import json
import logging
from pathlib import Path
from datetime import datetime
from configs.config import RESULTS_DIR, LOGS_DIR

from src.downloader import download_audio, DownloadError
from src.provenance_tracker import gate, update_manifest, ConsentError
from src.preprocessor import preprocess, PreprocessingError
from src.diarizer import load_pipeline as load_diarizer, diarize, DiarizationError
from src.emotion_tagger import load_emotion_model, tag_emotions, EmotionTaggingError

LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_already_processed(video_id: str) -> bool:
    annotation_path = RESULTS_DIR / f"{video_id}_annotations.json"
    return annotation_path.exists()


def run_single(url: str, diarizer, emotion_model) -> dict:
    video_id = None
    status = {"url": url, "success": False, "stage_failed": None, "error": None}

    try:
        logger.info(f"=== STAGE 1: DOWNLOAD | {url} ===")
        meta = download_audio(url)
        video_id = meta["video_id"]
        status["video_id"] = video_id

        if is_already_processed(video_id):
            logger.info(f"{video_id}: already processed, skipping")
            status["success"] = True
            status["skipped"] = True
            return status

        logger.info(f"=== STAGE 2: CONSENT GATE | {video_id} ===")
        provenance_record = gate(video_id)
        update_manifest(provenance_record)

        logger.info(f"=== STAGE 3: PREPROCESS | {video_id} ===")
        preprocess_result = preprocess(video_id, meta["raw_ext"])

        logger.info(f"=== STAGE 4: DIARIZE | {video_id} ===")
        diarization_output = diarize(
            video_id,
            diarizer,
            preprocess_result["audio"]["duration"]
        )

        logger.info(f"=== STAGE 5: EMOTION TAG | {video_id} ===")
        emotion_output = tag_emotions(video_id, diarization_output, emotion_model)

        logger.info(f"=== STAGE 6: ASSEMBLE OUTPUT | {video_id} ===")
        annotation = {
            "video_id": video_id,
            "source_url": url,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "provenance": provenance_record,
            "audio": preprocess_result["audio"],
            "diarization_validation": diarization_output["validation"],
            "emotion_stats": emotion_output["stats"],
            "segments": emotion_output["tagged_segments"],
        }

        annotation_path = RESULTS_DIR / f"{video_id}_annotations.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation, f, indent=2)

        logger.info(f"=== COMPLETE: {video_id} | {len(annotation['segments'])} segments ===")
        status["success"] = True

    except DownloadError as e:
        status["stage_failed"] = "download"
        status["error"] = str(e)
        logger.error(f"Download failed: {e}")

    except ConsentError as e:
        status["stage_failed"] = "consent_gate"
        status["error"] = str(e)
        logger.warning(f"Consent gate blocked: {e}")

    except PreprocessingError as e:
        status["stage_failed"] = "preprocess"
        status["error"] = str(e)
        logger.error(f"Preprocessing failed: {e}")

    except DiarizationError as e:
        status["stage_failed"] = "diarize"
        status["error"] = str(e)
        logger.error(f"Diarization failed: {e}")

    except EmotionTaggingError as e:
        status["stage_failed"] = "emotion_tag"
        status["error"] = str(e)
        logger.error(f"Emotion tagging failed: {e}")

    except Exception as e:
        status["stage_failed"] = "unknown"
        status["error"] = str(e)
        logger.exception(f"Unexpected error: {e}")

    return status


def run_batch(urls: list) -> list:
    logger.info(f"Loading models for batch of {len(urls)} URLs")
    diarizer = load_diarizer()
    emotion_model = load_emotion_model()

    results = []
    for i, url in enumerate(urls, 1):
        logger.info(f"Processing {i}/{len(urls)}: {url}")
        result = run_single(url, diarizer, emotion_model)
        results.append(result)

    summary_path = RESULTS_DIR / "batch_summary.json"
    summary = {
        "total": len(results),
        "succeeded": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "blocked_by_consent": sum(1 for r in results if r.get("stage_failed") == "consent_gate"),
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Batch complete: {summary['succeeded']}/{summary['total']} succeeded")
    return results
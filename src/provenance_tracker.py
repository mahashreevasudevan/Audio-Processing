import json
import logging
from pathlib import Path
from datetime import datetime
from configs.config import DATA_PROVENANCE, RESULTS_DIR, LICENCE_TIERS

logger = logging.getLogger(__name__)


class ConsentError(Exception):
    pass


def classify_licence(licence_string: str) -> str:
    if not licence_string:
        return "UNKNOWN"

    licence_lower = licence_string.lower()

    for tier, patterns in LICENCE_TIERS.items():
        for pattern in patterns:
            if pattern.lower() in licence_lower:
                return tier

    if "youtube" in licence_lower or "standard" in licence_lower:
        return "PROPRIETARY"

    return "UNKNOWN"


def build_provenance_record(video_id: str) -> dict:
    meta_path = DATA_PROVENANCE / f"{video_id}_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata found for {video_id}")

    with open(meta_path) as f:
        meta = json.load(f)

    licence_raw = meta.get("licence") or ""
    tier = classify_licence(licence_raw)
    gate_passed = tier == "OPEN"
    blocked_reason = None

    if tier == "PROPRIETARY":
        blocked_reason = f"Proprietary licence: {licence_raw or 'YouTube Standard'}"
    elif tier == "UNKNOWN":
        blocked_reason = "Licence information missing or unrecognised"
    elif tier == "RESTRICTED":
        blocked_reason = f"Restricted licence requires review: {licence_raw}"
        gate_passed = False

    record = {
        "video_id": video_id,
        "source_url": meta.get("source_url"),
        "uploader": meta.get("uploader"),
        "upload_date": meta.get("upload_date"),
        "licence_raw": licence_raw,
        "licence_tier": tier,
        "gate_passed": gate_passed,
        "blocked_reason": blocked_reason,
        "retrieved_at": meta.get("retrieved_at"),
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
    }

    record_path = DATA_PROVENANCE / f"{video_id}_provenance.json"
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


def gate(video_id: str) -> dict:
    record = build_provenance_record(video_id)
    tier = record["licence_tier"]

    if tier == "OPEN":
        logger.info(f"{video_id}: consent gate passed (licence: {record['licence_raw']})")
        return record

    elif tier == "RESTRICTED":
        logger.warning(f"{video_id}: RESTRICTED licence — flagged for manual review. Blocking.")
        raise ConsentError(f"{video_id} blocked: {record['blocked_reason']}")

    else:
        logger.error(f"{video_id}: BLOCKED — {record['blocked_reason']}")
        raise ConsentError(f"{video_id} blocked: {record['blocked_reason']}")


def update_manifest(record: dict):
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS_DIR / "provenance_manifest.csv"

    row = {
        "video_id": record["video_id"],
        "source_url": record["source_url"],
        "uploader": record["uploader"],
        "licence_tier": record["licence_tier"],
        "gate_passed": record["gate_passed"],
        "blocked_reason": record["blocked_reason"],
        "retrieved_at": record["retrieved_at"],
    }

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        df = df[df["video_id"] != record["video_id"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(manifest_path, index=False)
    logger.info(f"Provenance manifest updated: {manifest_path}")
import yt_dlp
import json
import logging
from pathlib import Path
from datetime import datetime
from configs.config import DATA_RAW, DATA_PROVENANCE, LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DownloadError(Exception):
    pass


def download_audio(url: str, max_retries: int = 3) -> dict:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROVENANCE.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(DATA_RAW / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
        "retries": max_retries,
        "ignoreerrors": False,
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}: downloading {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

            video_id = info.get("id")
            ext = info.get("ext")
            raw_path = DATA_RAW / f"{video_id}.{ext}"

            if not raw_path.exists():
                raise DownloadError(f"Download completed but file not found: {raw_path}")

            meta = {
                "video_id": video_id,
                "source_url": url,
                "title": info.get("title"),
                "uploader": info.get("uploader"),
                "uploader_id": info.get("uploader_id"),
                "upload_date": info.get("upload_date"),
                "licence": info.get("license"),
                "description": (info.get("description") or "")[:500],
                "duration": info.get("duration"),
                "view_count": info.get("view_count"),
                "retrieved_at": datetime.utcnow().isoformat() + "Z",
                "raw_file": str(raw_path),
                "raw_ext": ext,
            }

            meta_path = DATA_PROVENANCE / f"{video_id}_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"Downloaded {video_id} to {raw_path}")
            return meta

        except yt_dlp.utils.DownloadError as e:
            logger.warning(f"yt-dlp error on attempt {attempt}: {e}")
            if attempt == max_retries:
                raise DownloadError(f"Failed after {max_retries} attempts: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise DownloadError(str(e))
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_PROVENANCE = BASE_DIR / "data" / "provenance"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

HF_TOKEN = os.getenv("HF_TOKEN")

SAMPLE_RATE = 16000
CHANNELS = 1

LICENCE_TIERS = {
    "OPEN": ["creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/", "CC0", "cc0"],
    "RESTRICTED": ["creativecommons.org/licenses/by-nc/", "creativecommons.org/licenses/by-nd/"],
    "PROPRIETARY": ["youtube.com/t/terms"],
}

EMOTION_CONFIDENCE_THRESHOLD = 0.6
MIN_SEGMENT_DURATION = 1.0
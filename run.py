from src.pipeline import run_batch

URLS = [
    # CC licensed video we already tested
    "https://www.youtube.com/watch?v=CBLbDPmmx4M",
    # Add a regular YouTube video to test consent gate blocking
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
]

if __name__ == "__main__":
    results = run_batch(URLS)
    print("\n=== BATCH RESULTS ===")
    for r in results:
        status = "✅" if r["success"] else "❌"
        stage = r.get("stage_failed", "ok")
        vid = r.get("video_id", r["url"][:40])
        print(f"{status} {vid} — {stage}")
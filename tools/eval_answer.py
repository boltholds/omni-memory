import json, time, httpx, statistics
CASES = [
  {"q":"Where is Alice?", "expect_conflict": True},
  {"q":"List facts about Alice", "expect_conflict": False},
]
lat = []
hallus = 0
with httpx.Client(timeout=30) as s:
    for c in CASES:
        t0 = time.perf_counter()
        r = s.post("http://127.0.0.1:8000/answer", json={"q": c["q"], "lang":"en"})
        lat.append(time.perf_counter()-t0)
    m = s.get("http://127.0.0.1:8000/metrics").text
    if "qa_hallucinations_total" in m and " qa_hallucinations_total 0" not in m:
        hallus = 1
print(f"n={len(CASES)} p50={statistics.median(lat):.3f}s hallus_flag={hallus}")

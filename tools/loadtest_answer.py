import time
import httpx
import statistics
URL="http://127.0.0.1:8000/answer"
def one():
    with httpx.Client(timeout=30.0) as c:
        t0=time.perf_counter()
        r=c.post(URL,json={"q":"Where is Alice?"})
        r.raise_for_status()
        return (time.perf_counter()-t0)*1000
if __name__=="__main__":
    xs=[one() for _ in range(20)]
    print(f"n={len(xs)} p50={statistics.median(xs):.0f}ms max={max(xs):.0f}ms")

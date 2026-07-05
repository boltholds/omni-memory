from __future__ import annotations
import time
from collections import defaultdict

class Stats:
    def __init__(self) -> None:
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)  # key -> [ms,...]

    def inc(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def timeit(self, key: str):
        t0 = time.perf_counter()
        def stop():
            dur = int((time.perf_counter() - t0) * 1000)
            self.timers[key].append(dur)
            return dur
        return stop

    def snapshot(self):
        # p50/p95 оценим на лету
        import numpy as np
        out_t = {}
        for k, vals in self.timers.items():
            if not vals:
                continue
            arr = np.array(vals)
            out_t[k] = {
                "count": int(arr.size),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max()),
            }
        return {"counters": dict(self.counters), "timers": out_t}

stats = Stats()

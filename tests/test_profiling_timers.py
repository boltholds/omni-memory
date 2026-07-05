from omni_memory.profiling import timed

def test_timed_decorator_runs():
    calls=[]
    @timed("unit.test", slow_ms=0)
    def f(): calls.append(1)
    f()
    assert calls == [1]

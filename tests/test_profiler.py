from xpyrment.profiler import profile_execution

def test_profiler():
    def dummy_func(x):
        return x * 2
    res = profile_execution(dummy_func, 5)
    assert res == 10

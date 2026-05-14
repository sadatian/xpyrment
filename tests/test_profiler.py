from xpyrment.profiler import profile_execution

def test_profiler(capsys):
    def dummy_func(x):
        return x * 2
    res = profile_execution(dummy_func, 5)
    assert res == 10
    
    captured = capsys.readouterr()
    assert "Execution Profile for dummy_func" in captured.out
    assert "Time Elapsed" in captured.out
    assert "Peak Memory" in captured.out

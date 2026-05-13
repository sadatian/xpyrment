import time
import tracemalloc
from typing import Callable, Any

def profile_execution(func: Callable, *args, **kwargs) -> Any:
    """Profiles the execution time and peak memory of a given function."""
    tracemalloc.start()
    start_time = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Execution Profile for {func.__name__}:")
    print(f"  Time Elapsed: {end_time - start_time:.4f} seconds")
    print(f"  Peak Memory:  {peak / 10**6:.4f} MB")
    
    return result

import io
import json
import logging
import time
import pytest

from xpyrment.core.telemetry import configure_telemetry, get_logger, ExecutionProfiler


def test_json_formatter_and_configuration():
    """Asserts that telemetry configuration outputs valid, structured, parsable JSON strings."""
    stream = io.StringIO()
    logger = configure_telemetry(level=logging.INFO, stream=stream)

    # Log a standard message
    logger.info("Test event occurred")

    # Fetch stream content
    log_line = stream.getvalue().strip()
    assert log_line.startswith("{") and log_line.endswith("}")

    # Parse JSON
    parsed = json.loads(log_line)
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Test event occurred"
    assert "timestamp" in parsed
    assert parsed["logger"] == "xpyrment.telemetry"


def test_execution_profiler_context_manager():
    """Validates that ExecutionProfiler correctly profiles stage timing, memory peaks, and stage labels."""
    stream = io.StringIO()
    logger = configure_telemetry(level=logging.INFO, stream=stream)

    with ExecutionProfiler("inference_solve_stage", logger=logger):
        # Perform some dummy operations that consume time and memory
        time.sleep(0.05)
        dummy_list = [x for x in range(100000)]
        del dummy_list

    log_line = stream.getvalue().strip()
    parsed = json.loads(log_line)

    assert parsed["stage"] == "inference_solve_stage"
    assert parsed["status"] == "SUCCESS"
    assert parsed["duration_seconds"] >= 0.04
    assert parsed["peak_memory_kb"] >= 0.0
    assert "Profile completed for stage" in parsed["message"]


def test_execution_profiler_decorator():
    """Validates that the ExecutionProfiler behaves correctly when wrapping functions as a decorator."""
    stream = io.StringIO()
    logger = configure_telemetry(level=logging.INFO, stream=stream)

    @ExecutionProfiler("decorator_wrapped_stage", logger=logger)
    def my_heavy_function(a, b):
        time.sleep(0.02)
        return a + b

    result = my_heavy_function(10, 20)
    assert result == 30

    log_line = stream.getvalue().strip()
    parsed = json.loads(log_line)

    assert parsed["stage"] == "decorator_wrapped_stage"
    assert parsed["status"] == "SUCCESS"
    assert parsed["duration_seconds"] >= 0.01


def test_execution_profiler_exception_logging():
    """Asserts that ExecutionProfiler cleanly logs failed stages, attaches error info, and propagates standard exceptions."""
    stream = io.StringIO()
    logger = configure_telemetry(level=logging.ERROR, stream=stream)

    with pytest.raises(ZeroDivisionError):
        with ExecutionProfiler("failing_division_stage", logger=logger):
            val = 1 / 0

    log_line = stream.getvalue().strip()
    parsed = json.loads(log_line)

    assert parsed["stage"] == "failing_division_stage"
    assert parsed["status"] == "FAILED"
    assert parsed["error_type"] == "ZeroDivisionError"
    assert "division by zero" in parsed["error_message"].lower()

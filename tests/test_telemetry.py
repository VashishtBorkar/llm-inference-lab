from __future__ import annotations

import io
import tempfile
import time
import unittest
from pathlib import Path

from inference_lab.telemetry import (
    EventTimeline,
    NvidiaSmiCollector,
    parse_nvidia_smi_line,
)


class FakeProcess:
    def __init__(self, line: str) -> None:
        self.stdout = io.StringIO(line + "\n")
        self.stderr = io.StringIO("")
        self.terminated = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float):
        return 0

    def kill(self) -> None:
        self.terminated = True


class TelemetryTests(unittest.TestCase):
    def test_parses_supported_values_and_nulls(self) -> None:
        line = (
            "2026/08/19 12:00:00.000, 0, GPU-test, P0, 75, 99, 48, "
            "3000, 6144, 1455, 1455, 6000, 67.5, 0x4, Not Active, "
            "Not Active, Active, Not Active, Not Active, [N/A], Not Active"
        )
        sample = parse_nvidia_smi_line(line)
        self.assertEqual(sample["temperature_c"], 75.0)
        self.assertEqual(sample["sm_clock_mhz"], 1455.0)
        self.assertTrue(sample["limited_sw_power"])
        self.assertIsNone(sample["limited_hw_thermal"])

    def test_event_timeline_uses_run_relative_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            timeline = EventTimeline(path, run_id="run", origin_perf_ns=0)
            timeline.emit("phase_start", phase="measurement")
            contents = path.read_text(encoding="utf-8")
            self.assertIn('"event":"phase_start"', contents)
            self.assertIn('"run_id":"run"', contents)

    def test_collector_writes_sample_and_stops_process(self) -> None:
        line = (
            "2026/08/19 12:00:00.000, 0, GPU-test, P0, 75, 99, 48, "
            "3000, 6144, 1455, 1455, 6000, 67.5, 0x0, Not Active, "
            "Not Active, Not Active, Not Active, Not Active, Not Active, Not Active"
        )
        process = FakeProcess(line)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.jsonl"
            collector = NvidiaSmiCollector(
                path,
                run_id="run",
                origin_perf_ns=time.perf_counter_ns(),
                interval_ms=500,
                popen_factory=lambda *args, **kwargs: process,
            )
            collector.start()
            self.assertTrue(collector.wait_for_sample(1))
            collector.stop()
            self.assertTrue(process.terminated)
            self.assertEqual(collector.metadata()["sample_count"], 1)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()

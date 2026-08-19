from __future__ import annotations

import io
import tempfile
import threading
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
    def test_start_gate_uses_only_samples_observed_after_wait_begins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            collector = NvidiaSmiCollector(
                Path(temporary) / "gpu.jsonl",
                run_id="run",
                origin_perf_ns=time.perf_counter_ns(),
                interval_ms=500,
            )
            with collector._sample_condition:
                collector._samples.append(
                    {
                        "sample_sequence": 0,
                        "temperature_c": 40.0,
                        "gpu_utilization_pct": 0.0,
                    }
                )

            def add_new_samples() -> None:
                time.sleep(0.02)
                with collector._sample_condition:
                    collector._samples.extend(
                        [
                            {
                                "sample_sequence": 1,
                                "temperature_c": 70.0,
                                "gpu_utilization_pct": 0.0,
                            },
                            {
                                "sample_sequence": 2,
                                "temperature_c": 49.0,
                                "gpu_utilization_pct": 1.0,
                            },
                            {
                                "sample_sequence": 3,
                                "temperature_c": 48.0,
                                "gpu_utilization_pct": 2.0,
                            },
                        ]
                    )
                    collector._sample_condition.notify_all()

            producer = threading.Thread(target=add_new_samples)
            producer.start()
            result = collector.wait_for_start_gate(
                max_temperature_c=50.0,
                max_gpu_utilization_pct=5.0,
                consecutive_samples=2,
                timeout_seconds=1.0,
            )
            producer.join()

            self.assertEqual(result["first_new_sample_sequence"], 1)
            self.assertEqual(result["final_sample_sequence"], 3)
            self.assertEqual(result["evaluated_samples"], 3)

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

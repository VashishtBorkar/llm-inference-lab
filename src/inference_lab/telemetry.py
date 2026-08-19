from __future__ import annotations

import csv
import json
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any


class TelemetryError(RuntimeError):
    """Raised when required GPU telemetry cannot be collected."""


QUERY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "pstate",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "clocks_event_reasons.active",
    "clocks_event_reasons.gpu_idle",
    "clocks_event_reasons.applications_clocks_setting",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
)

NORMALIZED_FIELDS = (
    "gpu_index",
    "gpu_uuid",
    "performance_state",
    "temperature_c",
    "gpu_utilization_pct",
    "memory_io_utilization_pct",
    "memory_used_mib",
    "memory_total_mib",
    "graphics_clock_mhz",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "power_draw_w",
    "clock_event_reason_mask_hex",
    "limited_gpu_idle",
    "limited_application_clocks",
    "limited_sw_power",
    "limited_hw_slowdown",
    "limited_sw_thermal",
    "limited_hw_thermal",
    "limited_hw_power_brake",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _nullable(value: str) -> str | None:
    stripped = value.strip()
    if not stripped or stripped.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    return stripped


def _integer(value: str) -> int | None:
    parsed = _nullable(value)
    if parsed is None:
        return None
    try:
        return int(float(parsed))
    except ValueError:
        return None


def _floating(value: str) -> float | None:
    parsed = _nullable(value)
    if parsed is None:
        return None
    try:
        return float(parsed)
    except ValueError:
        return None


def _active(value: str) -> bool | None:
    parsed = _nullable(value)
    if parsed is None:
        return None
    lowered = parsed.lower()
    if lowered == "active":
        return True
    if lowered == "not active":
        return False
    return None


def parse_nvidia_smi_line(line: str) -> dict[str, Any]:
    values = next(csv.reader([line], skipinitialspace=True), [])
    if len(values) != len(QUERY_FIELDS):
        raise TelemetryError(
            f"nvidia-smi returned {len(values)} fields; expected {len(QUERY_FIELDS)}"
        )

    return {
        "nvidia_timestamp_local_raw": _nullable(values[0]),
        "gpu_index": _integer(values[1]),
        "gpu_uuid": _nullable(values[2]),
        "performance_state": _nullable(values[3]),
        "temperature_c": _floating(values[4]),
        "gpu_utilization_pct": _floating(values[5]),
        "memory_io_utilization_pct": _floating(values[6]),
        "memory_used_mib": _floating(values[7]),
        "memory_total_mib": _floating(values[8]),
        "graphics_clock_mhz": _floating(values[9]),
        "sm_clock_mhz": _floating(values[10]),
        "memory_clock_mhz": _floating(values[11]),
        "power_draw_w": _floating(values[12]),
        "clock_event_reason_mask_hex": _nullable(values[13]),
        "limited_gpu_idle": _active(values[14]),
        "limited_application_clocks": _active(values[15]),
        "limited_sw_power": _active(values[16]),
        "limited_hw_slowdown": _active(values[17]),
        "limited_sw_thermal": _active(values[18]),
        "limited_hw_thermal": _active(values[19]),
        "limited_hw_power_brake": _active(values[20]),
    }


class EventTimeline:
    def __init__(self, path: Path, *, run_id: str, origin_perf_ns: int) -> None:
        self.path = path
        self.run_id = run_id
        self.origin_perf_ns = origin_perf_ns
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, event: str, **details: Any) -> None:
        now = time.perf_counter_ns()
        record = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "sequence": self._sequence,
            "event": event,
            "offset_ms": (now - self.origin_perf_ns) / 1_000_000,
            "observed_at_utc": _utc_now(),
            **details,
        }
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            self._sequence += 1
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


class NvidiaSmiCollector:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        origin_perf_ns: int,
        interval_ms: int,
        popen_factory: Any = subprocess.Popen,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.origin_perf_ns = origin_perf_ns
        self.interval_ms = interval_ms
        self._popen_factory = popen_factory
        self._phase = "initializing"
        self._phase_lock = threading.Lock()
        self._sample_condition = threading.Condition()
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        query = ",".join(QUERY_FIELDS)
        self.command = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ]

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def _current_phase(self) -> str:
        with self._phase_lock:
            return self._phase

    def start(self) -> None:
        try:
            self._process = self._popen_factory(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except (FileNotFoundError, OSError) as exc:
            raise TelemetryError(f"Could not start nvidia-smi telemetry: {exc}") from exc

        self._thread = threading.Thread(
            target=self._read_loop,
            name="nvidia-smi-telemetry",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                for line in process.stdout:
                    if self._stopping.is_set():
                        break
                    if not line.strip():
                        continue
                    try:
                        parsed = parse_nvidia_smi_line(line)
                    except TelemetryError as exc:
                        self._errors.append(str(exc))
                        continue
                    now = time.perf_counter_ns()
                    sample = {
                        "schema_version": "1.0",
                        "run_id": self.run_id,
                        "sample_sequence": len(self._samples),
                        "sample_offset_ms": (now - self.origin_perf_ns) / 1_000_000,
                        "observed_at_utc": _utc_now(),
                        "phase": self._current_phase(),
                        **parsed,
                    }
                    handle.write(
                        json.dumps(sample, ensure_ascii=False, separators=(",", ":"))
                    )
                    handle.write("\n")
                    handle.flush()
                    with self._sample_condition:
                        self._samples.append(sample)
                        self._sample_condition.notify_all()
        except OSError as exc:
            self._errors.append(f"telemetry writer failed: {exc}")

    def wait_for_sample(self, timeout_seconds: float) -> bool:
        deadline = time.monotonic() + timeout_seconds
        with self._sample_condition:
            while not self._samples:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._sample_condition.wait(remaining)
            return True

    def stop(self) -> None:
        self._stopping.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if process is not None and process.stderr is not None:
            try:
                stderr = process.stderr.read().strip()
            except OSError:
                stderr = ""
            if stderr:
                self._errors.append(stderr)

    def metadata(self) -> dict[str, Any]:
        samples = list(self._samples)
        offsets = [float(sample["sample_offset_ms"]) for sample in samples]
        intervals = [right - left for left, right in pairwise(offsets)]
        total_values = len(samples) * len(NORMALIZED_FIELDS)
        missing_values = sum(
            sample.get(field) is None for sample in samples for field in NORMALIZED_FIELDS
        )
        supported = {
            field: any(sample.get(field) is not None for sample in samples)
            for field in NORMALIZED_FIELDS
        }
        return {
            "provider": "nvidia-smi",
            "command": self.command,
            "configured_interval_ms": self.interval_ms,
            "sample_count": len(samples),
            "median_interval_ms": statistics.median(intervals) if intervals else None,
            "max_sampling_gap_ms": max(intervals) if intervals else None,
            "missing_value_rate": (
                missing_values / total_values if total_values else None
            ),
            "field_support": supported,
            "errors": list(self._errors),
        }

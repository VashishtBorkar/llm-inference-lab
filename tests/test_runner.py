from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from inference_lab.models import (
    GenerationObservation,
    RunConfig,
    Scenario,
    StreamEventObservation,
    StreamTimingConfig,
)
from inference_lab.runner import StreamTimingError, run_benchmark


class FakeAdapter:
    name = "fake"

    def model_metadata(self, model: str):
        return {"name": model, "digest": "test-digest"}

    def generate(self, *, model: str, scenario: Scenario, keep_alive: str):
        start = time.perf_counter_ns()
        response = "synthetic response"
        return GenerationObservation(
            started_at_utc="2026-08-18T00:00:00+00:00",
            started_perf_ns=start,
            first_chunk_perf_ns=start + 10_000_000,
            first_content_perf_ns=start + 20_000_000,
            completed_perf_ns=start + 120_000_000,
            status="success",
            http_status=200,
            error_type=None,
            error_message=None,
            response_text=response,
            response_chars=len(response),
            response_sha256=hashlib.sha256(response.encode()).hexdigest(),
            stream_chunk_count=4,
            model=model,
            done_reason="stop",
            total_duration_ns=100_000_000,
            load_duration_ns=0,
            prompt_eval_count=10,
            prompt_eval_duration_ns=20_000_000,
            eval_count=4,
            eval_duration_ns=60_000_000,
        )


class TimingFakeAdapter(FakeAdapter):
    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        keep_alive: str,
        stream_timing=None,
        stream_event_callback=None,
    ):
        observation = super().generate(
            model=model, scenario=scenario, keep_alive=keep_alive
        )
        stream_events = tuple(
            StreamEventObservation(
                event_index=index + 1,
                received_perf_ns=observation.started_perf_ns + (index + 1) * 10_000_000,
                previous_event_delta_ns=10_000_000 if index else None,
                server_created_at=None,
                content_chars=1,
                thinking_chars=0,
                cumulative_content_chars=index + 1,
                cumulative_thinking_chars=0,
                selected_token_count=1,
                cumulative_selected_token_count=index + 1,
                done=False,
            )
            for index in range(4)
        )
        if stream_event_callback is not None:
            for event in stream_events:
                stream_event_callback(event)
        return replace(
            observation,
            stream_events=stream_events,
            stream_logprobs_requested=True,
        )


class MismatchedTimingAdapter(FakeAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        keep_alive: str,
        stream_timing=None,
        stream_event_callback=None,
    ):
        self.calls += 1
        observation = super().generate(
            model=model, scenario=scenario, keep_alive=keep_alive
        )
        event = StreamEventObservation(
            event_index=1,
            received_perf_ns=observation.started_perf_ns + 10_000_000,
            previous_event_delta_ns=None,
            server_created_at=None,
            content_chars=1,
            thinking_chars=0,
            cumulative_content_chars=1,
            cumulative_thinking_chars=0,
            selected_token_count=1,
            cumulative_selected_token_count=1,
            done=False,
        )
        if stream_event_callback is not None:
            stream_event_callback(event)
        return replace(
            observation,
            stream_events=(event,),
            stream_logprobs_requested=True,
        )


class FailedTimingAdapter(FakeAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        keep_alive: str,
        stream_timing=None,
        stream_event_callback=None,
    ):
        self.calls += 1
        observation = super().generate(
            model=model, scenario=scenario, keep_alive=keep_alive
        )
        return replace(
            observation,
            status="failed",
            error_type="SyntheticError",
            error_message="synthetic calibration failure",
            eval_count=None,
            eval_duration_ns=None,
        )


class RunnerTests(unittest.TestCase):
    def test_required_token_coverage_fails_when_warmup_request_fails(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workload = project_root / "workloads" / "smoke"
        adapter = FailedTimingAdapter()
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                model="test-model",
                workload_path=workload,
                output_root=Path(directory) / "runs",
                warmup=1,
                repetitions=1,
                concurrency=1,
                stream_timing=StreamTimingConfig(
                    enabled=True,
                    request_token_logprobs=True,
                    require_token_counts=True,
                    include_warmup=True,
                ),
            )
            with (
                patch("inference_lab.runner.collect_environment", return_value={}),
                self.assertRaisesRegex(
                    StreamTimingError, "calibration request failed"
                ),
            ):
                run_benchmark(
                    config=config,
                    adapter=adapter,
                    repo_root=project_root,
                )

        self.assertEqual(adapter.calls, 1)

    def test_required_token_coverage_fails_during_warmup(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workload = project_root / "workloads" / "smoke"
        adapter = MismatchedTimingAdapter()
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                model="test-model",
                workload_path=workload,
                output_root=Path(directory) / "runs",
                warmup=1,
                repetitions=1,
                concurrency=1,
                stream_timing=StreamTimingConfig(
                    enabled=True,
                    request_token_logprobs=True,
                    require_token_counts=True,
                    include_warmup=True,
                ),
            )
            with (
                patch("inference_lab.runner.collect_environment", return_value={}),
                self.assertRaisesRegex(StreamTimingError, "did not match"),
            ):
                run_benchmark(
                    config=config,
                    adapter=adapter,
                    repo_root=project_root,
                )

        self.assertEqual(adapter.calls, 1)

    def test_writes_privacy_safe_stream_timing_artifact_and_coverage(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workload = project_root / "workloads" / "smoke"
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                model="test-model",
                workload_path=workload,
                output_root=Path(directory) / "runs",
                warmup=1,
                repetitions=1,
                concurrency=1,
                stream_timing=StreamTimingConfig(
                    enabled=True,
                    request_token_logprobs=True,
                    require_token_counts=True,
                    include_warmup=True,
                ),
            )
            with patch("inference_lab.runner.collect_environment", return_value={}):
                result = run_benchmark(
                    config=config,
                    adapter=TimingFakeAdapter(),
                    repo_root=project_root,
                )

            stream_path = result.run_dir / "stream_events.jsonl"
            events = [
                json.loads(line)
                for line in stream_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(events), 24)
            self.assertTrue(
                all(event["token_count_class"] == "one_selected_token" for event in events)
            )
            self.assertTrue(all("content" not in event for event in events))
            self.assertEqual(
                result.manifest["stream_timing"][
                    "exact_eval_count_coverage_requests"
                ],
                6,
            )
            self.assertFalse(
                result.manifest["stream_timing"]["gpu_exact_token_timing"]
            )

    def test_warmup_output_limit_does_not_change_measured_scenarios(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workload = project_root / "workloads" / "smoke"
        adapter = FakeAdapter()
        observed_limits: list[int] = []
        original_generate = adapter.generate

        def generate(*, model: str, scenario: Scenario, keep_alive: str):
            observed_limits.append(scenario.generation["max_output_tokens"])
            return original_generate(
                model=model, scenario=scenario, keep_alive=keep_alive
            )

        adapter.generate = generate  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                model="test-model",
                workload_path=workload,
                output_root=Path(directory) / "runs",
                warmup=1,
                warmup_max_output_tokens=7,
                repetitions=1,
                concurrency=1,
            )
            with patch("inference_lab.runner.collect_environment", return_value={}):
                run_benchmark(config=config, adapter=adapter, repo_root=project_root)

        self.assertEqual(observed_limits[:3], [7, 7, 7])
        self.assertNotEqual(observed_limits[3:], [7, 7, 7])

    def test_writes_manifest_raw_records_and_warmup_free_summary(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workload = project_root / "workloads" / "smoke"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runs"
            config = RunConfig(
                model="test-model",
                workload_path=workload,
                output_root=output,
                warmup=1,
                repetitions=2,
                concurrency=1,
            )
            with patch("inference_lab.runner.collect_environment", return_value={"test": True}):
                result = run_benchmark(
                    config=config,
                    adapter=FakeAdapter(),
                    repo_root=project_root,
                )

            self.assertTrue((result.run_dir / "manifest.json").is_file())
            self.assertTrue((result.run_dir / "requests.jsonl").is_file())
            self.assertTrue((result.run_dir / "summary.json").is_file())
            lines = (result.run_dir / "requests.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 9)
            records = [json.loads(line) for line in lines]
            self.assertEqual(sum(item["is_warmup"] for item in records), 3)
            self.assertTrue(all(item["response_text"] is None for item in records))
            self.assertEqual(result.summary["overall"]["requests"], 6)
            self.assertEqual(result.summary["overall"]["successful"], 6)
            self.assertEqual(result.manifest["status"], "completed")

    def test_inter_request_delay_excludes_the_final_request(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        workload = project_root / "workloads" / "smoke"
        with tempfile.TemporaryDirectory() as directory:
            config = RunConfig(
                model="test-model",
                workload_path=workload,
                output_root=Path(directory) / "runs",
                warmup=0,
                repetitions=1,
                concurrency=1,
                inter_request_delay_seconds=2.0,
            )
            with (
                patch("inference_lab.runner.collect_environment", return_value={}),
                patch("inference_lab.runner.time.sleep") as sleep,
            ):
                result = run_benchmark(
                    config=config,
                    adapter=FakeAdapter(),
                    repo_root=project_root,
                )

            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(result.summary["timing"]["scheduled_idle_seconds"], 4.0)


if __name__ == "__main__":
    unittest.main()

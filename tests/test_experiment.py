from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_lab.cli import main
from inference_lab.experiment import ExperimentError, load_experiment, run_experiment
from inference_lab.models import GenerationObservation, Scenario


class FakeAdapter:
    name = "ollama"

    def __init__(self, **_: object) -> None:
        pass

    def model_metadata(self, model: str):
        return {"name": model, "digest": "fake"}

    def generate(self, *, model: str, scenario: Scenario, keep_alive: str):
        start = time.perf_counter_ns()
        response = "ok"
        return GenerationObservation(
            started_at_utc="2026-08-19T00:00:00+00:00",
            started_perf_ns=start,
            first_chunk_perf_ns=start + 1_000_000,
            first_content_perf_ns=start + 2_000_000,
            completed_perf_ns=start + 3_000_000,
            status="success",
            http_status=200,
            error_type=None,
            error_message=None,
            response_text=response,
            response_chars=len(response),
            response_sha256=hashlib.sha256(response.encode()).hexdigest(),
            stream_chunk_count=2,
            model=model,
            done_reason="stop",
            total_duration_ns=3_000_000,
            load_duration_ns=0,
            prompt_eval_count=4,
            prompt_eval_duration_ns=1_000_000,
            eval_count=2,
            eval_duration_ns=1_000_000,
        )


def _write_workload(root: Path) -> None:
    workload = root / "workloads" / "test"
    workload.mkdir(parents=True)
    (workload / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": "1.0",
                "bundle_id": "test",
                "bundle_version": "1.0.0",
                "scenario_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (workload / "scenarios.jsonl").write_text(
        json.dumps(
            {
                "scenario_id": "test",
                "task_type": "test",
                "workload_class": "test",
                "messages": [{"role": "user", "content": "test"}],
                "generation": {"max_output_tokens": 2},
                "validators": ["non_empty"],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_experiment(root: Path, *, duplicate: bool = False) -> Path:
    directory = root / "experiments" / "exp-test"
    directory.mkdir(parents=True)
    repeated = """
[[conditions]]
id = "baseline"
label = "Duplicate"
[conditions.run]
inter_request_delay_seconds = 0.0
""" if duplicate else ""
    (directory / "experiment.toml").write_text(
        f"""schema_version = "1.0"
[experiment]
id = "exp-test"
title = "Test"
question = "Does it work?"
hypothesis = "It should work."
[defaults]
engine = "ollama"
base_url = "http://127.0.0.1:11434"
model = "test-model"
workload = "workloads/test"
warmup = 0
repetitions = 1
concurrency = 1
timeout_seconds = 5.0
keep_alive = "5m"
capture_output = false
inter_request_delay_seconds = 0.0
[execution]
trials_per_condition = 1
condition_order = "fixed"
order_seed = 0
between_runs_seconds = 0.0
[telemetry]
enabled = false
required = false
interval_ms = 500
pre_roll_seconds = 0.0
post_roll_seconds = 0.0
[[conditions]]
id = "baseline"
label = "Baseline"
[conditions.run]
inter_request_delay_seconds = 0.0
{repeated}
""",
        encoding="utf-8",
    )
    return directory


class ExperimentTests(unittest.TestCase):
    def test_loads_and_hashes_valid_specification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_workload(root)
            directory = _write_experiment(root)
            spec = load_experiment(directory, repo_root=root)
            self.assertEqual(spec.experiment_id, "exp-test")
            self.assertEqual(len(spec.specification_sha256), 64)
            self.assertEqual(spec.conditions[0].condition_id, "baseline")

    def test_rejects_duplicate_condition_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_workload(root)
            directory = _write_experiment(root, duplicate=True)
            with self.assertRaisesRegex(ExperimentError, "Duplicate condition"):
                load_experiment(directory, repo_root=root)

    def test_validate_command_does_not_construct_ollama_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_workload(root)
            directory = _write_experiment(root)
            with (
                patch("inference_lab.experiment.OllamaAdapter") as adapter,
                patch("pathlib.Path.cwd", return_value=root),
            ):
                result = main(["experiment", "validate", str(directory)])
            self.assertEqual(result, 0)
            adapter.assert_not_called()

    def test_run_expands_schedule_and_writes_experiment_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_workload(root)
            directory = _write_experiment(root)
            spec = load_experiment(directory, repo_root=root)
            with (
                patch("inference_lab.experiment.OllamaAdapter", FakeAdapter),
                patch(
                    "inference_lab.runner.collect_environment", return_value={"test": True}
                ),
            ):
                result = run_experiment(spec)
            self.assertEqual(result.index["status"], "completed")
            self.assertEqual(len(result.runs), 1)
            manifest = result.runs[0].manifest
            self.assertEqual(manifest["experiment"]["experiment_id"], "exp-test")
            self.assertEqual(manifest["experiment"]["condition_id"], "baseline")
            self.assertEqual(manifest["experiment"]["trial_number"], 1)


if __name__ == "__main__":
    unittest.main()

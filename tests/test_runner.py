from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_lab.models import GenerationObservation, RunConfig, Scenario
from inference_lab.runner import run_benchmark


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


class RunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

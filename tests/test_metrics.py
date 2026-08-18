from __future__ import annotations

import unittest

from inference_lab.metrics import derive_client_metrics, metric_stats, percentile
from inference_lab.models import GenerationObservation


class MetricsTests(unittest.TestCase):
    def test_derives_client_latency_and_decode_metrics(self) -> None:
        start = 1_000_000_000
        observation = GenerationObservation(
            started_at_utc="2026-08-18T00:00:00+00:00",
            started_perf_ns=start,
            first_chunk_perf_ns=start + 50_000_000,
            first_content_perf_ns=start + 100_000_000,
            completed_perf_ns=start + 1_100_000_000,
            status="success",
            http_status=200,
            error_type=None,
            error_message=None,
            response_text="hello",
            response_chars=5,
            response_sha256="unused",
            stream_chunk_count=11,
            model="test",
            done_reason="stop",
            total_duration_ns=1_000_000_000,
            load_duration_ns=0,
            prompt_eval_count=20,
            prompt_eval_duration_ns=200_000_000,
            eval_count=11,
            eval_duration_ns=900_000_000,
        )

        metrics = derive_client_metrics(observation)

        self.assertEqual(metrics["client_time_to_first_chunk_ms"], 50.0)
        self.assertEqual(metrics["client_ttft_ms"], 100.0)
        self.assertEqual(metrics["client_e2e_ms"], 1100.0)
        self.assertEqual(metrics["client_decode_ms"], 1000.0)
        self.assertEqual(metrics["client_tpot_ms"], 100.0)
        self.assertEqual(metrics["client_decode_tokens_per_second"], 10.0)

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)

    def test_p95_is_withheld_for_too_few_samples(self) -> None:
        stats = metric_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["median"], 2.0)
        self.assertIsNone(stats["p95"])


if __name__ == "__main__":
    unittest.main()


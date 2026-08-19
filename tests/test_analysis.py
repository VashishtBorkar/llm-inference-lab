from __future__ import annotations

import unittest

from inference_lab.analysis import _join_request_gpu


class AnalysisTests(unittest.TestCase):
    def test_joins_only_gpu_samples_inside_request_interval(self) -> None:
        request = {"started_offset_ms": 100.0, "completed_offset_ms": 300.0}
        samples = [
            {"sample_offset_ms": 50.0, "temperature_c": 50.0},
            {
                "sample_offset_ms": 150.0,
                "temperature_c": 70.0,
                "sm_clock_mhz": 1400.0,
                "gpu_utilization_pct": 90.0,
                "power_draw_w": 60.0,
                "limited_sw_thermal": True,
            },
            {
                "sample_offset_ms": 250.0,
                "temperature_c": 80.0,
                "sm_clock_mhz": 1300.0,
                "gpu_utilization_pct": 95.0,
                "power_draw_w": 65.0,
                "limited_sw_thermal": False,
            },
            {"sample_offset_ms": 350.0, "temperature_c": 90.0},
        ]

        joined = _join_request_gpu(request, samples)

        self.assertEqual(joined["gpu_sample_count"], 2)
        self.assertEqual(joined["gpu_temperature_mean_c"], 75.0)
        self.assertEqual(joined["gpu_sm_clock_min_mhz"], 1300.0)
        self.assertTrue(joined["gpu_limited_sw_thermal_seen"])
        self.assertTrue(joined["gpu_limiter_seen"])


if __name__ == "__main__":
    unittest.main()

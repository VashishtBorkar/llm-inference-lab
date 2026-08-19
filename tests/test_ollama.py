from __future__ import annotations

import json
import unittest
from typing import Self
from unittest.mock import patch

from inference_lab.engines.ollama import OllamaAdapter
from inference_lab.models import Scenario, StreamTimingConfig


class FakeStreamResponse:
    status = 200

    def __init__(self, events: list[dict[str, object]]) -> None:
        self.lines = [(json.dumps(event) + "\n").encode("utf-8") for event in events]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class OllamaAdapterTests(unittest.TestCase):
    def test_records_privacy_safe_selected_token_event_counts(self) -> None:
        events = [
            {
                "created_at": "2026-08-19T00:00:00Z",
                "message": {"content": "A"},
                "logprobs": [{"token": "secret-a", "logprob": -0.1}],
                "done": False,
            },
            {
                "created_at": "2026-08-19T00:00:00.010Z",
                "message": {"content": "BC"},
                "logprobs": [
                    {"token": "secret-b", "logprob": -0.2},
                    {"token": "secret-c", "logprob": -0.3},
                ],
                "done": False,
            },
            {
                "created_at": "2026-08-19T00:00:00.020Z",
                "message": {"content": ""},
                "done": True,
                "eval_count": 3,
                "eval_duration": 30_000_000,
            },
        ]
        opened_requests = []

        def opener(request, timeout):
            opened_requests.append(request)
            return FakeStreamResponse(events)

        scenario = Scenario(
            scenario_id="timing",
            task_type="chat",
            workload_class="decode",
            messages=({"role": "user", "content": "Generate."},),
            generation={"max_output_tokens": 3, "context_window": 8192},
            validators=("non_empty",),
        )
        adapter = OllamaAdapter(opener=opener)

        with patch(
            "inference_lab.engines.ollama.time.perf_counter_ns",
            side_effect=[
                1_000_000_000,
                1_010_000_000,
                1_025_000_000,
                1_030_000_000,
            ],
        ):
            observation = adapter.generate(
                model="test-model",
                scenario=scenario,
                keep_alive="5m",
                stream_timing=StreamTimingConfig(enabled=True),
            )

        payload = json.loads(opened_requests[0].data.decode("utf-8"))
        self.assertTrue(payload["logprobs"])
        self.assertEqual(payload["top_logprobs"], 0)
        self.assertEqual(payload["options"]["num_ctx"], 8192)
        self.assertEqual(
            [event.selected_token_count for event in observation.stream_events],
            [1, 2, None],
        )
        self.assertEqual(
            observation.stream_events[1].previous_event_delta_ns, 15_000_000
        )
        serialized = repr(observation.stream_events)
        self.assertNotIn("secret-a", serialized)
        self.assertNotIn("logprob", serialized)

    def test_streams_chat_and_collects_final_usage(self) -> None:
        events = [
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": "Hello"},
                "done": False,
            },
            {
                "model": "test-model",
                "message": {"role": "assistant", "content": " world"},
                "done": True,
                "done_reason": "stop",
                "total_duration": 200_000_000,
                "load_duration": 10_000_000,
                "prompt_eval_count": 7,
                "prompt_eval_duration": 50_000_000,
                "eval_count": 2,
                "eval_duration": 100_000_000,
            },
        ]
        opened_requests = []

        def opener(request, timeout):
            opened_requests.append((request, timeout))
            return FakeStreamResponse(events)

        scenario = Scenario(
            scenario_id="test",
            task_type="chat",
            workload_class="control",
            messages=({"role": "user", "content": "Hello"},),
            generation={"temperature": 0, "seed": 42, "max_output_tokens": 8},
            validators=("non_empty",),
        )
        adapter = OllamaAdapter(opener=opener, timeout_seconds=12)

        with patch(
            "inference_lab.engines.ollama.time.perf_counter_ns",
            side_effect=[1_000_000_000, 1_050_000_000, 1_200_000_000],
        ):
            observation = adapter.generate(
                model="test-model", scenario=scenario, keep_alive="5m"
            )

        self.assertEqual(observation.status, "success")
        self.assertEqual(observation.response_text, "Hello world")
        self.assertEqual(observation.first_content_perf_ns, 1_050_000_000)
        self.assertEqual(observation.completed_perf_ns, 1_200_000_000)
        self.assertEqual(observation.prompt_eval_count, 7)
        self.assertEqual(observation.eval_count, 2)
        self.assertEqual(opened_requests[0][1], 12)
        request_payload = json.loads(opened_requests[0][0].data.decode("utf-8"))
        self.assertTrue(request_payload["stream"])
        self.assertEqual(request_payload["options"]["num_predict"], 8)

    def test_preserves_midstream_error_as_failed_observation(self) -> None:
        events = [
            {"message": {"content": "partial"}, "done": False},
            {"error": "model runner stopped"},
        ]
        adapter = OllamaAdapter(opener=lambda request, timeout: FakeStreamResponse(events))
        scenario = Scenario(
            scenario_id="test",
            task_type="chat",
            workload_class="control",
            messages=({"role": "user", "content": "Hello"},),
            generation={"max_output_tokens": 8},
            validators=("non_empty",),
        )

        observation = adapter.generate(
            model="test-model", scenario=scenario, keep_alive="5m"
        )

        self.assertEqual(observation.status, "failed")
        self.assertEqual(observation.error_message, "model runner stopped")
        self.assertEqual(observation.response_text, "partial")


if __name__ == "__main__":
    unittest.main()

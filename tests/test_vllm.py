from __future__ import annotations

import json
import unittest
from typing import Self
from unittest.mock import patch

from pathlib import Path

from inference_lab.engines.factory import (
    create_adapter,
    default_engine_options,
    normalize_engine_options,
)
from inference_lab.engines.vllm import VllmAdapter, VllmError
from inference_lab.models import RunConfig, Scenario, StreamTimingConfig


class FakeResponse:
    status = 200

    def __init__(self, *, payload: dict | None = None, lines: list[str] | None = None):
        self.payload = payload
        self.lines = lines or []

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        return iter((line + "\n").encode("utf-8") for line in self.lines)


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="chat-test",
        task_type="chat",
        workload_class="control",
        messages=({"role": "user", "content": "Hello"},),
        generation={
            "temperature": 0,
            "seed": 42,
            "max_output_tokens": 8,
            "context_window": 4096,
        },
        validators=("non_empty",),
    )


class VllmAdapterTests(unittest.TestCase):
    def test_engine_options_reject_settings_for_another_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported vllm engine options"):
            normalize_engine_options("vllm", {"device": "cuda"})

    def test_factory_uses_vllm_default_url(self) -> None:
        adapter = create_adapter(
            RunConfig(
                engine="vllm",
                model="test-model",
                workload_path=Path("workloads/smoke"),
                output_root=Path("runs"),
                timeout_seconds=12,
            )
        )
        self.assertIsInstance(adapter, VllmAdapter)
        self.assertEqual(adapter.base_url, "http://127.0.0.1:8000")
        self.assertEqual(
            default_engine_options("ollama")["base_url"],
            "http://127.0.0.1:11434",
        )

    def test_model_metadata_selects_served_model(self) -> None:
        adapter = VllmAdapter(
            opener=lambda request, timeout: FakeResponse(
                payload={"object": "list", "data": [{"id": "test-model"}]}
            )
        )
        self.assertEqual(adapter.model_metadata("test-model")["id"], "test-model")
        with self.assertRaisesRegex(VllmError, "not served"):
            adapter.model_metadata("missing")

    def test_streams_chat_collects_usage_and_token_event_counts(self) -> None:
        lines = [
            "data: " + json.dumps(
                {
                    "created": 1,
                    "choices": [
                        {
                            "delta": {"content": "Hello"},
                            "finish_reason": None,
                            "logprobs": {"content": [{"token": "private"}]},
                        }
                    ],
                }
            ),
            "data: " + json.dumps(
                {
                    "created": 2,
                    "choices": [
                        {
                            "delta": {"content": " world"},
                            "finish_reason": "stop",
                            "logprobs": {"content": [{"token": "secret"}]},
                        }
                    ],
                }
            ),
            "data: " + json.dumps(
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }
            ),
            "data: [DONE]",
        ]
        requests = []

        def opener(request, timeout):
            requests.append(request)
            return FakeResponse(lines=lines)

        adapter = VllmAdapter(opener=opener, api_key="do-not-record")
        with patch(
            "inference_lab.engines.vllm.time.perf_counter_ns",
            side_effect=[
                1_000_000_000,
                1_010_000_000,
                1_020_000_000,
                1_025_000_000,
                1_030_000_000,
            ],
        ):
            observation = adapter.generate(
                model="test-model",
                scenario=_scenario(),
                stream_timing=StreamTimingConfig(enabled=True),
            )

        self.assertEqual(observation.status, "success")
        self.assertEqual(observation.response_text, "Hello world")
        self.assertEqual(observation.prompt_eval_count, 7)
        self.assertEqual(observation.eval_count, 2)
        self.assertEqual(
            [event.selected_token_count for event in observation.stream_events],
            [1, 1, None],
        )
        self.assertNotIn("private", repr(observation.stream_events))
        payload = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(payload["max_tokens"], 8)
        self.assertNotIn("context_window", payload)
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertEqual(requests[0].headers["Authorization"], "Bearer do-not-record")


if __name__ == "__main__":
    unittest.main()

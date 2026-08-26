from __future__ import annotations

import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from inference_lab.engines.factory import create_adapter
from inference_lab.engines.pytorch_reference import (
    PyTorchReferenceAdapter,
    PyTorchReferenceError,
)
from inference_lab.models import RunConfig, Scenario, StreamTimingConfig


class FakeTensor:
    def __init__(self, values, shape, *, dtype="int64", device="cpu"):
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def item(self):
        return self.values[0]


class FakeLogits:
    def __getitem__(self, key):
        return self


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def synchronize():
        raise AssertionError("CPU inference should not synchronize CUDA")


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.next_tokens = iter([4, 5, 2])

    @staticmethod
    def inference_mode():
        return nullcontext()

    def argmax(self, logits, *, dim, keepdim):
        return FakeTensor([next(self.next_tokens)], (1, 1))

    @staticmethod
    def ones(shape, *, dtype, device):
        return FakeTensor([1], shape, dtype=dtype, device=device)

    @staticmethod
    def cat(tensors, *, dim):
        left, right = tensors
        return FakeTensor(
            left.values + right.values,
            (left.shape[0], left.shape[1] + right.shape[1]),
            dtype=left.dtype,
            device=left.device,
        )


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.rendered_messages = messages
        return "rendered prompt"

    def __call__(self, prompt, *, return_tensors):
        return {
            "input_ids": FakeTensor([10, 11, 12], (1, 3)),
            "attention_mask": FakeTensor([1, 1, 1], (1, 3)),
        }

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        return "".join({4: "A", 5: "B", 2: ""}[token] for token in token_ids)


class FakeModel:
    dtype = "float32"
    config = SimpleNamespace(model_type="fake-causal-lm")

    def __init__(self):
        self.input_lengths = []

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        past_key_values=None,
    ):
        self.input_lengths.append(input_ids.shape[-1])
        return SimpleNamespace(
            logits=FakeLogits(),
            past_key_values=(len(self.input_lengths),),
        )


def _scenario(**generation_overrides) -> Scenario:
    generation = {"temperature": 0, "seed": 42, "max_output_tokens": 8}
    generation.update(generation_overrides)
    return Scenario(
        scenario_id="learning",
        task_type="chat",
        workload_class="control",
        messages=({"role": "user", "content": "Explain."},),
        generation=generation,
        validators=("non_empty",),
    )


class PyTorchReferenceTests(unittest.TestCase):
    def test_factory_constructs_reference_adapter(self) -> None:
        adapter = create_adapter(
            RunConfig(
                engine="pytorch_reference",
                model="fake/model",
                workload_path=Path("workloads/smoke"),
                output_root=Path("runs"),
                timeout_seconds=1,
                engine_options={"device": "cpu", "dtype": "float32"},
            )
        )
        self.assertIsInstance(adapter, PyTorchReferenceAdapter)
        self.assertEqual(adapter.requested_device, "cpu")

    def test_rejects_sampling_to_keep_reference_loop_simple(self) -> None:
        with self.assertRaisesRegex(PyTorchReferenceError, "greedy generation only"):
            PyTorchReferenceAdapter._validate_generation(
                _scenario(temperature=0.7, top_p=0.9)
            )

    def test_runs_prefill_then_one_token_decode_steps_with_kv_cache(self) -> None:
        fake_torch = FakeTorch()
        tokenizer = FakeTokenizer()
        model = FakeModel()
        adapter = PyTorchReferenceAdapter(
            device="cpu",
            dtype="float32",
            torch_module=fake_torch,
            tokenizer=tokenizer,
            model_instance=model,
        )
        observed_events = []
        with patch(
            "inference_lab.engines.pytorch_reference.time.perf_counter_ns",
            side_effect=[100, 110, 130, 131, 150, 170, 180],
        ):
            observation = adapter.generate(
                model="fake/model",
                scenario=_scenario(),
                stream_timing=StreamTimingConfig(enabled=True),
                stream_event_callback=observed_events.append,
            )

        self.assertEqual(observation.status, "success")
        self.assertEqual(observation.response_text, "AB")
        self.assertEqual(observation.prompt_eval_count, 3)
        self.assertEqual(observation.eval_count, 3)
        self.assertEqual(observation.done_reason, "stop")
        self.assertEqual(model.input_lengths, [3, 1, 1])
        self.assertEqual(len(observed_events), 3)
        self.assertTrue(observed_events[-1].done)
        self.assertTrue(
            all(event.selected_token_count == 1 for event in observed_events)
        )


if __name__ == "__main__":
    unittest.main()

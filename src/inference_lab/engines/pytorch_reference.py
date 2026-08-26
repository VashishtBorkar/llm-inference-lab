from __future__ import annotations

import hashlib
import threading
import time
from datetime import UTC, datetime
from typing import Any

from inference_lab.engines.base import EngineCapabilities, StreamEventCallback
from inference_lab.models import (
    GenerationObservation,
    Scenario,
    StreamEventObservation,
    StreamTimingConfig,
)


class PyTorchReferenceError(RuntimeError):
    """Raised when the intentionally small PyTorch backend cannot run."""


class PyTorchReferenceAdapter:
    """Readable batch-size-one greedy decoding with a visible KV-cache loop.

    Transformers loads the tokenizer, model definition, and weights. Generation is
    kept here instead of delegated to ``model.generate`` so prefill and repeated
    one-token decode steps remain visible.
    """

    name = "pytorch_reference"
    capabilities = EngineCapabilities(
        server_token_counts=True,
        server_timing=True,
        selected_token_counts=True,
    )
    configuration_notes = {
        "purpose": "learning reference; not a production serving engine",
        "batch_size": 1,
        "generation_policy": "greedy argmax",
        "kv_cache": True,
        "concurrency": "requests are serialized around one model instance",
        "portable_generation_settings": ["max_output_tokens"],
        "unsupported_generation_settings": [
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "context_window",
            "think",
        ],
        "ignored_generation_settings": ["seed"],
    }

    def __init__(
        self,
        *,
        device: str = "auto",
        dtype: str = "auto",
        torch_module: Any | None = None,
        tokenizer: Any | None = None,
        model_instance: Any | None = None,
    ) -> None:
        self.requested_device = device
        self.requested_dtype = dtype
        self._torch = torch_module
        self._tokenizer = tokenizer
        self._model = model_instance
        self._loaded_model_id: str | None = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def _load(self, model: str) -> None:
        if self._model is not None and self._tokenizer is not None:
            if self._loaded_model_id not in {None, model}:
                raise PyTorchReferenceError(
                    "one adapter instance cannot load more than one model"
                )
            self._loaded_model_id = model
            self._device = self.requested_device
            return
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise PyTorchReferenceError(
                "PyTorch reference dependencies are missing. Install with "
                "`pip install -e '.[pytorch]'`."
            ) from exc

        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_by_name = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.requested_dtype not in dtype_by_name:
            raise PyTorchReferenceError(
                "dtype must be auto, float32, float16, or bfloat16"
            )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype_by_name[self.requested_dtype],
        )
        self._model.to(device)
        self._model.eval()
        self._loaded_model_id = model
        self._device = device

    def _synchronize(self) -> None:
        if (
            self._device is not None
            and self._device.startswith("cuda")
            and self._torch.cuda.is_available()
        ):
            self._torch.cuda.synchronize()

    def model_metadata(self, model: str) -> dict[str, Any]:
        self._load(model)
        config = getattr(self._model, "config", None)
        return {
            "id": model,
            "device": self._device,
            "dtype": str(getattr(self._model, "dtype", self.requested_dtype)),
            "architecture": (
                type(self._model).__name__ if self._model is not None else None
            ),
            "transformers_model_type": getattr(config, "model_type", None),
        }

    @staticmethod
    def _validate_generation(scenario: Scenario) -> int:
        if scenario.response_format != "text":
            raise PyTorchReferenceError(
                "the learning backend supports text responses only"
            )
        generation = scenario.generation
        unsupported = [
            name
            for name in (
                "top_p",
                "top_k",
                "stop",
                "context_window",
                "think",
            )
            if name in generation
        ]
        temperature = generation.get("temperature", 0)
        if temperature not in {0, 0.0}:
            unsupported.append("temperature")
        if unsupported:
            raise PyTorchReferenceError(
                "the learning backend supports greedy generation only; unsupported "
                f"settings: {', '.join(sorted(set(unsupported)))}"
            )
        value = generation.get("max_output_tokens")
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PyTorchReferenceError("max_output_tokens must be a positive integer")
        return value

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        stream_timing: StreamTimingConfig | None = None,
        stream_event_callback: StreamEventCallback | None = None,
    ) -> GenerationObservation:
        started_at_utc = datetime.now(UTC).isoformat()
        started_perf_ns = time.perf_counter_ns()
        first_token_perf_ns: int | None = None
        first_content_perf_ns: int | None = None
        completed_perf_ns = started_perf_ns
        prefill_duration_ns: int | None = None
        decode_duration_ns: int | None = None
        generated_ids: list[int] = []
        stream_events: list[StreamEventObservation] = []
        status = "failed"
        error_type: str | None = None
        error_message: str | None = None
        done_reason: str | None = None
        prompt_tokens: int | None = None

        try:
            max_output_tokens = self._validate_generation(scenario)
            self._load(model)
            with self._lock, self._torch.inference_mode():
                # 1. Turn structured chat messages into the exact model prompt.
                prompt = self._tokenizer.apply_chat_template(
                    list(scenario.messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # 2. Tokenize once and move the prompt tensors to the model device.
                encoded = self._tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self._device)
                attention_mask = encoded["attention_mask"].to(self._device)
                prompt_tokens = int(input_ids.shape[-1])

                # 3. Prefill processes every prompt token and creates the KV cache.
                self._synchronize()
                prefill_started_ns = time.perf_counter_ns()
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                next_token = self._torch.argmax(
                    outputs.logits[:, -1, :], dim=-1, keepdim=True
                )
                past_key_values = outputs.past_key_values
                self._synchronize()
                first_token_perf_ns = time.perf_counter_ns()
                prefill_duration_ns = first_token_perf_ns - prefill_started_ns

                # 4. Decode one token at a time, reusing the cached keys and values.
                previous_token_perf_ns: int | None = None
                cumulative_chars = 0
                for token_index in range(max_output_tokens):
                    token_id = int(next_token.item())
                    generated_ids.append(token_id)
                    token_text = self._tokenizer.decode(
                        [token_id], skip_special_tokens=True
                    )
                    token_perf_ns = time.perf_counter_ns()
                    if token_text and first_content_perf_ns is None:
                        first_content_perf_ns = token_perf_ns
                    cumulative_chars += len(token_text)
                    eos_token_id = self._tokenizer.eos_token_id
                    reached_eos = (
                        eos_token_id is not None and token_id == eos_token_id
                    )
                    reached_limit = token_index + 1 == max_output_tokens
                    event = StreamEventObservation(
                        event_index=token_index + 1,
                        received_perf_ns=token_perf_ns,
                        previous_event_delta_ns=(
                            token_perf_ns - previous_token_perf_ns
                            if previous_token_perf_ns is not None
                            else None
                        ),
                        server_created_at=None,
                        content_chars=len(token_text),
                        thinking_chars=0,
                        cumulative_content_chars=cumulative_chars,
                        cumulative_thinking_chars=0,
                        selected_token_count=1,
                        cumulative_selected_token_count=token_index + 1,
                        done=reached_eos or reached_limit,
                    )
                    stream_events.append(event)
                    if stream_event_callback is not None:
                        stream_event_callback(event)
                    previous_token_perf_ns = token_perf_ns

                    if reached_eos:
                        done_reason = "stop"
                        break
                    if reached_limit:
                        done_reason = "length"
                        break

                    # The attention mask grows, but only the newest token enters
                    # the model; earlier states come from past_key_values.
                    one = self._torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = self._torch.cat(
                        (attention_mask, one), dim=-1
                    )
                    outputs = self._model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    next_token = self._torch.argmax(
                        outputs.logits[:, -1, :], dim=-1, keepdim=True
                    )
                    past_key_values = outputs.past_key_values
                    self._synchronize()

                completed_perf_ns = time.perf_counter_ns()
                decode_duration_ns = completed_perf_ns - first_token_perf_ns
                status = "success"
        except Exception as exc:  # noqa: BLE001 - preserve a failed request record
            completed_perf_ns = time.perf_counter_ns()
            error_type = type(exc).__name__
            error_message = str(exc)

        response_text = (
            self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            if self._tokenizer is not None
            else ""
        )
        return GenerationObservation(
            started_at_utc=started_at_utc,
            started_perf_ns=started_perf_ns,
            first_chunk_perf_ns=first_token_perf_ns,
            first_content_perf_ns=first_content_perf_ns,
            completed_perf_ns=completed_perf_ns,
            status=status,
            http_status=None,
            error_type=error_type,
            error_message=error_message,
            response_text=response_text,
            response_chars=len(response_text),
            response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            stream_chunk_count=len(generated_ids),
            model=model,
            done_reason=done_reason,
            total_duration_ns=completed_perf_ns - started_perf_ns,
            load_duration_ns=None,
            prompt_eval_count=prompt_tokens,
            prompt_eval_duration_ns=prefill_duration_ns,
            eval_count=len(generated_ids),
            eval_duration_ns=decode_duration_ns,
            stream_events=tuple(stream_events),
            stream_logprobs_requested=False,
        )

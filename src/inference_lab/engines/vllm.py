from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from inference_lab.engines.base import EngineCapabilities, StreamEventCallback
from inference_lab.models import (
    GenerationObservation,
    Scenario,
    StreamEventObservation,
    StreamTimingConfig,
)


class VllmError(RuntimeError):
    """Raised when a vLLM OpenAI-compatible server cannot satisfy a request."""


class VllmAdapter:
    """Adapter for vLLM's OpenAI-compatible HTTP server."""

    name = "vllm"
    capabilities = EngineCapabilities(
        server_token_counts=True,
        selected_token_counts=True,
        structured_output=True,
    )
    configuration_notes = {
        "portable_generation_settings": [
            "temperature",
            "top_p",
            "top_k",
            "seed",
            "max_output_tokens",
            "stop",
        ],
        "unsupported_generation_settings": ["context_window", "think"],
        "server_lifecycle": "managed outside the benchmark harness",
    }

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 300.0,
        api_key: str | None = None,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key
        self._opener = opener or urllib.request.urlopen

    def _headers(self, *, content_type: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _http_error_message(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict) and error.get("message"):
                    return str(error["message"])
                if error:
                    return str(error)
            return body.strip() or str(exc)
        except (OSError, json.JSONDecodeError):
            return str(exc)

    def model_metadata(self, model: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/models", headers=self._headers(), method="GET"
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise VllmError(self._http_error_message(exc)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise VllmError(f"Cannot reach vLLM at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise VllmError(f"vLLM returned invalid model JSON: {exc.msg}") from exc
        models = payload.get("data", []) if isinstance(payload, dict) else []
        for candidate in models:
            if isinstance(candidate, dict) and candidate.get("id") == model:
                return candidate
        available = sorted(
            str(candidate["id"])
            for candidate in models
            if isinstance(candidate, dict) and candidate.get("id")
        )
        suffix = f" Available models: {', '.join(available)}" if available else ""
        raise VllmError(f"vLLM model '{model}' is not served.{suffix}")

    @staticmethod
    def _payload(
        model: str,
        scenario: Scenario,
        stream_timing: StreamTimingConfig | None,
    ) -> dict[str, Any]:
        generation = scenario.generation
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(scenario.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        portable = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "max_output_tokens": "max_tokens",
            "stop": "stop",
        }
        for portable_name, vllm_name in portable.items():
            if portable_name in generation:
                payload[vllm_name] = generation[portable_name]
        if scenario.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        elif scenario.response_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": scenario.scenario_id.replace("-", "_"),
                    "schema": scenario.response_schema,
                },
            }
        if (
            stream_timing
            and stream_timing.enabled
            and stream_timing.request_token_logprobs
        ):
            payload["logprobs"] = True
            payload["top_logprobs"] = 0
        return payload

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        stream_timing: StreamTimingConfig | None = None,
        stream_event_callback: StreamEventCallback | None = None,
    ) -> GenerationObservation:
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(self._payload(model, scenario, stream_timing)).encode(
                "utf-8"
            ),
            headers=self._headers(content_type=True),
            method="POST",
        )
        started_at_utc = datetime.now(UTC).isoformat()
        started_perf_ns = time.perf_counter_ns()
        completed_perf_ns = started_perf_ns
        first_chunk_perf_ns: int | None = None
        first_content_perf_ns: int | None = None
        http_status: int | None = None
        status = "failed"
        error_type: str | None = None
        error_message: str | None = None
        response_parts: list[str] = []
        chunks = 0
        finish_reason: str | None = None
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        stream_events: list[StreamEventObservation] = []
        previous_event_perf_ns: int | None = None
        cumulative_content_chars = 0
        cumulative_selected_token_count = 0
        timing_enabled = bool(stream_timing and stream_timing.enabled)

        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                http_status = getattr(response, "status", 200)
                for raw_line in response:
                    event_perf_ns = time.perf_counter_ns()
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        raise VllmError("stream event was not an SSE data record")
                    data = line[5:].strip()
                    if data == "[DONE]":
                        completed_perf_ns = event_perf_ns
                        status = "success"
                        break
                    payload = json.loads(data)
                    if not isinstance(payload, dict):
                        raise VllmError("stream event was not a JSON object")
                    if payload.get("error"):
                        raise VllmError(str(payload["error"]))
                    if first_chunk_perf_ns is None:
                        first_chunk_perf_ns = event_perf_ns
                    chunks += 1
                    choices = payload.get("choices") or []
                    choice = (
                        choices[0]
                        if choices and isinstance(choices[0], dict)
                        else {}
                    )
                    delta = choice.get("delta") or {}
                    content = delta.get("content", "") if isinstance(delta, dict) else ""
                    if not isinstance(content, str):
                        content = str(content)
                    if content and first_content_perf_ns is None:
                        first_content_perf_ns = event_perf_ns
                    if content:
                        response_parts.append(content)
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                    usage = payload.get("usage")
                    if isinstance(usage, dict):
                        if isinstance(usage.get("prompt_tokens"), int):
                            prompt_tokens = usage["prompt_tokens"]
                        if isinstance(usage.get("completion_tokens"), int):
                            output_tokens = usage["completion_tokens"]

                    if timing_enabled:
                        logprobs = choice.get("logprobs")
                        selected_count: int | None = None
                        if isinstance(logprobs, dict) and isinstance(
                            logprobs.get("content"), list
                        ):
                            selected_count = len(logprobs["content"])
                        cumulative_content_chars += len(content)
                        cumulative_selected_token_count += selected_count or 0
                        event = StreamEventObservation(
                            event_index=len(stream_events) + 1,
                            received_perf_ns=event_perf_ns,
                            previous_event_delta_ns=(
                                event_perf_ns - previous_event_perf_ns
                                if previous_event_perf_ns is not None
                                else None
                            ),
                            server_created_at=(
                                str(payload["created"])
                                if payload.get("created") is not None
                                else None
                            ),
                            content_chars=len(content),
                            thinking_chars=0,
                            cumulative_content_chars=cumulative_content_chars,
                            cumulative_thinking_chars=0,
                            selected_token_count=selected_count,
                            cumulative_selected_token_count=(
                                cumulative_selected_token_count
                            ),
                            done=choice.get("finish_reason") is not None,
                        )
                        stream_events.append(event)
                        if stream_event_callback:
                            stream_event_callback(event)
                        previous_event_perf_ns = event_perf_ns
                else:
                    completed_perf_ns = time.perf_counter_ns()
            if status != "success":
                raise VllmError("stream ended without a [DONE] event")
        except urllib.error.HTTPError as exc:
            completed_perf_ns = time.perf_counter_ns()
            http_status = exc.code
            error_type = type(exc).__name__
            error_message = self._http_error_message(exc)
        except (
            urllib.error.URLError,
            TimeoutError,
            VllmError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as exc:
            completed_perf_ns = time.perf_counter_ns()
            error_type = type(exc).__name__
            error_message = str(exc)
        except Exception as exc:  # noqa: BLE001 - retain failure record
            completed_perf_ns = time.perf_counter_ns()
            error_type = type(exc).__name__
            error_message = str(exc)

        response_text = "".join(response_parts)
        return GenerationObservation(
            started_at_utc=started_at_utc,
            started_perf_ns=started_perf_ns,
            first_chunk_perf_ns=first_chunk_perf_ns,
            first_content_perf_ns=first_content_perf_ns,
            completed_perf_ns=completed_perf_ns,
            status=status,
            http_status=http_status,
            error_type=error_type,
            error_message=error_message,
            response_text=response_text,
            response_chars=len(response_text),
            response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            stream_chunk_count=chunks,
            model=model,
            done_reason=finish_reason,
            total_duration_ns=None,
            load_duration_ns=None,
            prompt_eval_count=prompt_tokens,
            prompt_eval_duration_ns=None,
            eval_count=output_tokens,
            eval_duration_ns=None,
            stream_events=tuple(stream_events),
            stream_logprobs_requested=bool(
                timing_enabled
                and stream_timing
                and stream_timing.request_token_logprobs
            ),
        )

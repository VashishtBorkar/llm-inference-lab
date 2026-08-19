from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from inference_lab.engines.base import StreamEventCallback
from inference_lab.models import (
    GenerationObservation,
    Scenario,
    StreamEventObservation,
    StreamTimingConfig,
)


class OllamaError(RuntimeError):
    """Raised when Ollama cannot satisfy a benchmark request."""


class OllamaAdapter:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 300.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def _request_json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaError(f"Cannot reach Ollama at {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON from {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise OllamaError(f"Ollama returned an unexpected response from {path}")
        return payload

    def model_metadata(self, model: str) -> dict[str, Any]:
        payload = self._request_json("/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama /api/tags response did not include a model list")
        for candidate in models:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("name") == model or candidate.get("model") == model:
                return candidate
        available = sorted(
            str(candidate.get("name"))
            for candidate in models
            if isinstance(candidate, dict) and candidate.get("name")
        )
        suffix = f" Available models: {', '.join(available)}" if available else ""
        raise OllamaError(f"Ollama model '{model}' is not installed.{suffix}")

    @staticmethod
    def _payload(
        model: str,
        scenario: Scenario,
        keep_alive: str,
        stream_timing: StreamTimingConfig | None = None,
    ) -> dict[str, Any]:
        generation = scenario.generation
        options: dict[str, Any] = {}
        portable_options = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "max_output_tokens": "num_predict",
            "context_window": "num_ctx",
            "stop": "stop",
        }
        for portable_name, ollama_name in portable_options.items():
            if portable_name in generation:
                options[ollama_name] = generation[portable_name]

        payload: dict[str, Any] = {
            "model": model,
            "messages": list(scenario.messages),
            "stream": True,
            "keep_alive": keep_alive,
            "options": options,
        }
        if scenario.response_format == "json":
            payload["format"] = "json"
        elif scenario.response_format == "json_schema":
            payload["format"] = scenario.response_schema
        if "think" in generation:
            payload["think"] = generation["think"]
        if (
            stream_timing is not None
            and stream_timing.enabled
            and stream_timing.request_token_logprobs
        ):
            # Ollama's top_logprobs=0 returns only the selected token. The
            # adapter retains only how many selected tokens each event covers;
            # token text, bytes, and probability values are intentionally
            # discarded.
            payload["logprobs"] = True
            payload["top_logprobs"] = 0
        return payload

    @staticmethod
    def _error_message_from_http_error(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])
            return body.strip() or str(exc)
        except (OSError, json.JSONDecodeError):
            return str(exc)

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        keep_alive: str,
        stream_timing: StreamTimingConfig | None = None,
        stream_event_callback: StreamEventCallback | None = None,
    ) -> GenerationObservation:
        request_body = json.dumps(
            self._payload(model, scenario, keep_alive, stream_timing),
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=request_body,
            headers={
                "Accept": "application/x-ndjson",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        started_at_utc = datetime.now(UTC).isoformat()
        started_perf_ns = time.perf_counter_ns()
        first_chunk_perf_ns: int | None = None
        first_content_perf_ns: int | None = None
        completed_perf_ns = started_perf_ns
        http_status: int | None = None
        error_type: str | None = None
        error_message: str | None = None
        status = "failed"
        chunks = 0
        response_parts: list[str] = []
        final_payload: dict[str, Any] = {}
        stream_events: list[StreamEventObservation] = []
        previous_event_perf_ns: int | None = None
        cumulative_content_chars = 0
        cumulative_thinking_chars = 0
        cumulative_selected_token_count = 0
        timing_enabled = bool(stream_timing is not None and stream_timing.enabled)

        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                http_status = getattr(response, "status", 200)
                for raw_line in response:
                    event_perf_ns = time.perf_counter_ns()
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    if first_chunk_perf_ns is None:
                        first_chunk_perf_ns = event_perf_ns
                    chunks += 1
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise OllamaError("stream event was not a JSON object")
                    if payload.get("error"):
                        raise OllamaError(str(payload["error"]))

                    message = payload.get("message", {})
                    if not isinstance(message, dict):
                        message = {}
                    content = message.get("content", "")
                    thinking = message.get("thinking", "")
                    if not isinstance(content, str):
                        content = str(content)
                    if not isinstance(thinking, str):
                        thinking = str(thinking)
                    if (content or thinking) and first_content_perf_ns is None:
                        first_content_perf_ns = event_perf_ns
                    if content:
                        response_parts.append(content)

                    if timing_enabled:
                        if "logprobs" not in payload:
                            selected_token_count: int | None = None
                        else:
                            raw_logprobs = payload["logprobs"]
                            if not isinstance(raw_logprobs, list):
                                raise OllamaError(
                                    "stream event logprobs field was not an array"
                                )
                            selected_token_count = len(raw_logprobs)
                        cumulative_content_chars += len(content)
                        cumulative_thinking_chars += len(thinking)
                        cumulative_selected_token_count += selected_token_count or 0
                        stream_event = StreamEventObservation(
                            event_index=len(stream_events) + 1,
                            received_perf_ns=event_perf_ns,
                            previous_event_delta_ns=(
                                event_perf_ns - previous_event_perf_ns
                                if previous_event_perf_ns is not None
                                else None
                            ),
                            server_created_at=(
                                payload["created_at"]
                                if isinstance(payload.get("created_at"), str)
                                else None
                            ),
                            content_chars=len(content),
                            thinking_chars=len(thinking),
                            cumulative_content_chars=cumulative_content_chars,
                            cumulative_thinking_chars=cumulative_thinking_chars,
                            selected_token_count=selected_token_count,
                            cumulative_selected_token_count=(
                                cumulative_selected_token_count
                            ),
                            done=payload.get("done") is True,
                        )
                        stream_events.append(stream_event)
                        if stream_event_callback is not None:
                            stream_event_callback(stream_event)
                        previous_event_perf_ns = event_perf_ns

                    if payload.get("done") is True:
                        final_payload = payload
                        completed_perf_ns = event_perf_ns
                        status = "success"
                        break
                else:
                    completed_perf_ns = time.perf_counter_ns()

            if status != "success":
                raise OllamaError("stream ended without a final done event")
        except urllib.error.HTTPError as exc:
            completed_perf_ns = time.perf_counter_ns()
            http_status = exc.code
            error_type = type(exc).__name__
            error_message = self._error_message_from_http_error(exc)
        except (urllib.error.URLError, TimeoutError) as exc:
            completed_perf_ns = time.perf_counter_ns()
            error_type = type(exc).__name__
            error_message = str(exc)
        except (OllamaError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            completed_perf_ns = time.perf_counter_ns()
            error_type = type(exc).__name__
            error_message = str(exc)
        except Exception as exc:  # noqa: BLE001 - preserve a record for adapter failures
            completed_perf_ns = time.perf_counter_ns()
            error_type = type(exc).__name__
            error_message = str(exc)

        response_text = "".join(response_parts)

        def optional_int(field_name: str) -> int | None:
            value = final_payload.get(field_name)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

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
            model=str(final_payload.get("model") or model),
            done_reason=(
                str(final_payload["done_reason"])
                if final_payload.get("done_reason") is not None
                else None
            ),
            total_duration_ns=optional_int("total_duration"),
            load_duration_ns=optional_int("load_duration"),
            prompt_eval_count=optional_int("prompt_eval_count"),
            prompt_eval_duration_ns=optional_int("prompt_eval_duration"),
            eval_count=optional_int("eval_count"),
            eval_duration_ns=optional_int("eval_duration"),
            stream_events=tuple(stream_events),
            stream_logprobs_requested=bool(
                timing_enabled
                and stream_timing is not None
                and stream_timing.request_token_logprobs
            ),
        )

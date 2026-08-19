from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from inference_lab.models import (
    GenerationObservation,
    Scenario,
    StreamEventObservation,
    StreamTimingConfig,
)

StreamEventCallback = Callable[[StreamEventObservation], None]


class EngineAdapter(Protocol):
    name: str

    def model_metadata(self, model: str) -> dict[str, Any]: ...

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        keep_alive: str,
        stream_timing: StreamTimingConfig | None = None,
        stream_event_callback: StreamEventCallback | None = None,
    ) -> GenerationObservation: ...

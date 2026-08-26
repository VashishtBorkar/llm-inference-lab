from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from inference_lab.models import (
    GenerationObservation,
    Scenario,
    StreamEventObservation,
    StreamTimingConfig,
)

StreamEventCallback = Callable[[StreamEventObservation], None]


@dataclass(frozen=True)
class EngineCapabilities:
    """Features an adapter can expose without inventing measurements."""

    streaming: bool = True
    server_token_counts: bool = False
    server_timing: bool = False
    selected_token_counts: bool = False
    structured_output: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


class EngineAdapter(Protocol):
    name: str
    capabilities: EngineCapabilities
    configuration_notes: dict[str, Any]

    def model_metadata(self, model: str) -> dict[str, Any]: ...

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        stream_timing: StreamTimingConfig | None = None,
        stream_event_callback: StreamEventCallback | None = None,
    ) -> GenerationObservation: ...

from __future__ import annotations

from typing import Any, Protocol

from inference_lab.models import GenerationObservation, Scenario


class EngineAdapter(Protocol):
    name: str

    def model_metadata(self, model: str) -> dict[str, Any]: ...

    def generate(
        self,
        *,
        model: str,
        scenario: Scenario,
        keep_alive: str,
    ) -> GenerationObservation: ...


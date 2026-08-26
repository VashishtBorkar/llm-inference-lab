from __future__ import annotations

from inference_lab.engines.base import EngineAdapter
from inference_lab.engines.ollama import OllamaAdapter
from inference_lab.engines.pytorch_reference import PyTorchReferenceAdapter
from inference_lab.engines.vllm import VllmAdapter
from inference_lab.models import RunConfig

ENGINE_NAMES = ("ollama", "vllm", "pytorch_reference")


def default_engine_options(engine: str) -> dict[str, str]:
    if engine == "ollama":
        return {
            "base_url": "http://127.0.0.1:11434",
            "keep_alive": "5m",
        }
    if engine == "vllm":
        return {"base_url": "http://127.0.0.1:8000"}
    if engine == "pytorch_reference":
        return {"device": "auto", "dtype": "auto"}
    raise ValueError(f"Unsupported engine: {engine}")


def normalize_engine_options(
    engine: str, options: dict[str, object] | None
) -> dict[str, str]:
    normalized = default_engine_options(engine)
    supplied = options or {}
    unknown = set(supplied) - set(normalized)
    if unknown:
        raise ValueError(
            f"Unsupported {engine} engine options: {', '.join(sorted(unknown))}"
        )
    for name, value in supplied.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{engine} engine option '{name}' must be a string")
        normalized[name] = value.strip()
    if engine == "pytorch_reference" and normalized["dtype"] not in {
        "auto",
        "float32",
        "float16",
        "bfloat16",
    }:
        raise ValueError(
            "pytorch_reference dtype must be auto, float32, float16, or bfloat16"
        )
    return normalized


def create_adapter(
    config: RunConfig,
    *,
    api_key: str | None = None,
) -> EngineAdapter:
    options = normalize_engine_options(config.engine, config.engine_options)
    if config.engine == "ollama":
        return OllamaAdapter(
            base_url=options["base_url"],
            timeout_seconds=config.timeout_seconds,
            keep_alive=options["keep_alive"],
        )
    if config.engine == "vllm":
        return VllmAdapter(
            base_url=options["base_url"],
            timeout_seconds=config.timeout_seconds,
            api_key=api_key,
        )
    if config.engine == "pytorch_reference":
        return PyTorchReferenceAdapter(
            device=options["device"], dtype=options["dtype"]
        )
    raise ValueError(f"Unsupported engine: {config.engine}")

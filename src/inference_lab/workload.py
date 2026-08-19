from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from inference_lab.models import Scenario, WorkloadBundle


class WorkloadError(ValueError):
    """Raised when a workload bundle is missing or invalid."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkloadError(f"Missing workload file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkloadError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _require_string(value: Any, field_name: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkloadError(f"{source}: '{field_name}' must be a non-empty string")
    return value


def _load_response_schema(root: Path, schema_ref: Any, source: str) -> dict[str, Any] | None:
    if schema_ref is None:
        return None
    ref = _require_string(schema_ref, "response_schema_ref", source)
    schema_path = (root / ref).resolve()
    try:
        schema_path.relative_to(root.resolve())
    except ValueError as exc:
        raise WorkloadError(f"{source}: response schema must stay inside the bundle") from exc
    schema = _load_json(schema_path)
    if not isinstance(schema, dict):
        raise WorkloadError(f"{source}: response schema must contain a JSON object")
    return schema


def _parse_scenario(data: Any, root: Path, source: str) -> Scenario:
    if not isinstance(data, dict):
        raise WorkloadError(f"{source}: each scenario must be a JSON object")

    scenario_id = _require_string(data.get("scenario_id"), "scenario_id", source)
    task_type = _require_string(data.get("task_type"), "task_type", source)
    workload_class = _require_string(data.get("workload_class"), "workload_class", source)

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise WorkloadError(f"{source}: 'messages' must be a non-empty array")
    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(raw_messages):
        message_source = f"{source} message {index}"
        if not isinstance(raw_message, dict):
            raise WorkloadError(f"{message_source}: message must be an object")
        role = _require_string(raw_message.get("role"), "role", message_source)
        content = _require_string(raw_message.get("content"), "content", message_source)
        messages.append({"role": role, "content": content})

    generation = data.get("generation", {})
    if not isinstance(generation, dict):
        raise WorkloadError(f"{source}: 'generation' must be an object")
    if "max_output_tokens" not in generation:
        raise WorkloadError(f"{source}: generation.max_output_tokens is required")
    max_output_tokens = generation["max_output_tokens"]
    if not isinstance(max_output_tokens, int) or isinstance(max_output_tokens, bool) or max_output_tokens < 1:
        raise WorkloadError(f"{source}: generation.max_output_tokens must be a positive integer")
    context_window = generation.get("context_window")
    if context_window is not None and (
        not isinstance(context_window, int)
        or isinstance(context_window, bool)
        or context_window < 1
    ):
        raise WorkloadError(
            f"{source}: generation.context_window must be a positive integer"
        )

    raw_validators = data.get("validators", ["non_empty"])
    if not isinstance(raw_validators, list) or not all(
        isinstance(item, str) and item for item in raw_validators
    ):
        raise WorkloadError(f"{source}: 'validators' must be an array of strings")

    response_format = data.get("response_format", "text")
    if response_format not in {"text", "json", "json_schema"}:
        raise WorkloadError(
            f"{source}: response_format must be 'text', 'json', or 'json_schema'"
        )
    response_schema = _load_response_schema(root, data.get("response_schema_ref"), source)
    if response_format == "json_schema" and response_schema is None:
        raise WorkloadError(f"{source}: json_schema output requires response_schema_ref")

    validation = data.get("validation", {})
    if not isinstance(validation, dict):
        raise WorkloadError(f"{source}: 'validation' must be an object")

    tags = data.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
        raise WorkloadError(f"{source}: 'tags' must be an array of strings")

    data_classification = data.get("data_classification", "public")
    if data_classification not in {"public", "private_local"}:
        raise WorkloadError(
            f"{source}: data_classification must be 'public' or 'private_local'"
        )

    return Scenario(
        scenario_id=scenario_id,
        task_type=task_type,
        workload_class=workload_class,
        messages=tuple(messages),
        generation=dict(generation),
        validators=tuple(raw_validators),
        response_format=response_format,
        response_schema=response_schema,
        validation=dict(validation),
        tags=tuple(tags),
        data_classification=data_classification,
    )


def _bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_workload(path: Path) -> WorkloadBundle:
    root = path.resolve()
    if not root.is_dir():
        raise WorkloadError(f"Workload bundle is not a directory: {root}")

    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise WorkloadError("manifest.json must contain a JSON object")
    for field_name in ("format_version", "bundle_id", "bundle_version"):
        _require_string(manifest.get(field_name), field_name, "manifest.json")

    scenarios_path = root / "scenarios.jsonl"
    try:
        lines = scenarios_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise WorkloadError(f"Missing workload file: {scenarios_path}") from exc

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        source = f"scenarios.jsonl line {line_number}"
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkloadError(f"{source}: invalid JSON: {exc.msg}") from exc
        scenario = _parse_scenario(data, root, source)
        if scenario.scenario_id in seen_ids:
            raise WorkloadError(f"{source}: duplicate scenario_id '{scenario.scenario_id}'")
        seen_ids.add(scenario.scenario_id)
        scenarios.append(scenario)

    if not scenarios:
        raise WorkloadError("scenarios.jsonl must contain at least one scenario")
    declared_count = manifest.get("scenario_count")
    if declared_count is not None and declared_count != len(scenarios):
        raise WorkloadError(
            f"manifest scenario_count is {declared_count}, but loaded {len(scenarios)} scenarios"
        )

    return WorkloadBundle(
        root=root,
        manifest=manifest,
        scenarios=tuple(scenarios),
        content_sha256=_bundle_hash(root),
    )

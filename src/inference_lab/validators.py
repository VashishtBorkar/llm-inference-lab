from __future__ import annotations

import json
from typing import Any

from inference_lab.models import Scenario


def validate_response(scenario: Scenario, response_text: str) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    parsed_json: Any = None
    parse_attempted = False

    for validator in scenario.validators:
        if validator == "non_empty":
            passed = bool(response_text.strip())
            results[validator] = {"passed": passed}
            continue

        if validator in {"json_valid", "required_fields_present"}:
            if not parse_attempted:
                parse_attempted = True
                try:
                    parsed_json = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    parsed_json = None
                    parse_error = f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
                else:
                    parse_error = None

            if validator == "json_valid":
                results[validator] = {
                    "passed": parsed_json is not None,
                    "error": parse_error,
                }
                continue

            required_fields = scenario.validation.get("required_fields", [])
            if not isinstance(required_fields, list) or not all(
                isinstance(item, str) for item in required_fields
            ):
                results[validator] = {
                    "passed": False,
                    "error": "validation.required_fields must be an array of strings",
                }
            elif not isinstance(parsed_json, dict):
                results[validator] = {
                    "passed": False,
                    "error": parse_error or "response is not a JSON object",
                }
            else:
                missing = [field for field in required_fields if field not in parsed_json]
                results[validator] = {"passed": not missing, "missing": missing}
            continue

        if validator == "exact_match":
            expected = scenario.validation.get("exact_text")
            if not isinstance(expected, str):
                results[validator] = {
                    "passed": False,
                    "error": "validation.exact_text must be a string",
                }
            else:
                results[validator] = {
                    "passed": response_text.strip() == expected.strip()
                }
            continue

        results[validator] = {
            "passed": False,
            "error": f"unknown validator: {validator}",
        }

    return results


def validators_passed(results: dict[str, dict[str, Any]]) -> bool:
    return all(result.get("passed") is True for result in results.values())


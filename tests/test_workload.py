from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference_lab.workload import WorkloadError, load_workload


class WorkloadTests(unittest.TestCase):
    def _bundle(self, root: Path, scenarios: list[dict[str, object]]) -> None:
        manifest = {
            "format_version": "1.0",
            "bundle_id": "test",
            "bundle_version": "1.0.0",
            "scenario_count": len(scenarios),
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "scenarios.jsonl").write_text(
            "\n".join(json.dumps(item) for item in scenarios) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _scenario(scenario_id: str) -> dict[str, object]:
        return {
            "scenario_id": scenario_id,
            "task_type": "chat",
            "workload_class": "control",
            "messages": [{"role": "user", "content": "Hello"}],
            "generation": {"max_output_tokens": 8},
            "validators": ["non_empty"],
        }

    def test_loads_a_valid_bundle_and_hashes_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root, [self._scenario("one")])

            workload = load_workload(root)

            self.assertEqual(workload.manifest["bundle_id"], "test")
            self.assertEqual(workload.scenarios[0].scenario_id, "one")
            self.assertEqual(len(workload.content_sha256), 64)

    def test_rejects_duplicate_scenario_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root, [self._scenario("same"), self._scenario("same")])

            with self.assertRaisesRegex(WorkloadError, "duplicate scenario_id"):
                load_workload(root)

    def test_requires_positive_output_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario = self._scenario("bad")
            scenario["generation"] = {"max_output_tokens": 0}
            self._bundle(root, [scenario])

            with self.assertRaisesRegex(WorkloadError, "positive integer"):
                load_workload(root)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = (
    REPO_ROOT / "experiments" / "exp-006-ollama-prefill-kernel-scaling" / "profile.py"
)
PROFILE_CONFIG = PROFILE_SCRIPT.with_name("profile.toml")


def _load_profile_module() -> ModuleType:
    module_name = "exp006_nsight_profile"
    spec = importlib.util.spec_from_file_location(module_name, PROFILE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {PROFILE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class NsightProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_profile_module()
        cls.config = cls.module.load_config(PROFILE_CONFIG)

    def test_profile_matrix_covers_short_peak_and_long_contexts(self) -> None:
        self.assertEqual(
            self.config.prompt_tokens,
            (128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384),
        )
        self.assertEqual(self.config.pilot_prompt_tokens, (2048, 16384))

    def test_every_length_has_distinct_warmup_and_profile_scenarios(self) -> None:
        pairs = self.module.load_scenario_pairs(self.config)

        self.assertEqual(set(pairs), set(self.config.prompt_tokens))
        for prompt_tokens, scenarios in pairs.items():
            self.assertEqual(set(scenarios), {"warmup", "profile"})
            self.assertNotEqual(
                scenarios["warmup"].messages,
                scenarios["profile"].messages,
                f"{prompt_tokens}-token warmup would populate the measured prefix",
            )

    def test_nsys_command_delays_collection_until_after_warmup(self) -> None:
        command = self.module.build_nsys_command(
            self.config,
            session_name="test_session",
            output_prefix=Path("runs/test/nsight"),
        )

        self.assertIn("--start-later=true", command)
        self.assertIn("--cuda-graph-trace=node", command)
        self.assertIn("--sample=none", command)
        self.assertEqual(command[-2:], ["ollama", "serve"])


if __name__ == "__main__":
    unittest.main()

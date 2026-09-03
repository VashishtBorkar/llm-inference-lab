from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = (
    REPO_ROOT / "experiments" / "exp-006-ollama-prefill-kernel-scaling" / "analysis.py"
)


def _load_analysis_module() -> ModuleType:
    module_name = "exp006_nsight_analysis"
    spec = importlib.util.spec_from_file_location(module_name, ANALYSIS_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {ANALYSIS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class NsightAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_analysis_module()

    def test_retained_prefix_uses_first_cache_value_for_measured_request(self) -> None:
        log = """
slot | new prompt, n_ctx_slot = 16384
slot | cached n_tokens = 0
slot | cached n_tokens = 1024
slot | cached n_tokens = 2048
slot | new prompt, n_ctx_slot = 16384
slot | cached n_tokens = 24
slot | cached n_tokens = 1048
slot | cached n_tokens = 2072
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "ollama.log"
            log_path.write_text(log, encoding="utf-8")

            retained = self.module._retained_prefix_tokens(log_path)

        self.assertEqual(retained, 24)

    def test_decode_attention_launches_are_excluded(self) -> None:
        event = self.module.KernelEvent
        events = []
        for layer in range(28):
            base = layer * 100
            events.extend(
                event(base + i, base + i + 1, "void mul_mat_q<q12>", 84)
                for i in range(3)
            )
            events.append(event(base + 10, base + 11, "flash_attn_ext_f16", 1))
            events.append(event(base + 20, base + 21, "void mul_mat_q<q12>", 84))
            if layer < 27:
                events.extend(
                    event(base + i, base + i + 1, "void mul_mat_q<q12>", 1184)
                    for i in (30, 31)
                )
                events.append(
                    event(
                        base + 40,
                        base + 41,
                        "unary_gated_op_kernel<&op_silu>",
                        1,
                    )
                )
                events.append(
                    event(base + 50, base + 51, "void mul_mat_q<q14>", 84)
                )
        decode_attention = [
            event(3000 + i, 3001 + i, "flash_attn_ext_f16", 1)
            for i in range(28)
        ]

        families = self.module._family_events(
            events + decode_attention,
            chunk_count=1,
        )

        self.assertEqual([len(family) for family in families], [28, 81, 112, 27])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / "corpora"
    / "wikitext-2-raw-v1"
    / "wiki.train.raw"
)
CORPUS_SHA256 = "6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57"
CORPUS_ARCHIVE_SHA256 = (
    "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
)
CONTEXT_WINDOW = 32768
OUTPUT_TOKENS = 8
TAIL_INSTRUCTION = "\n\nReply with only OK."
TARGET_TOKENS = (128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384)
CORPUS_START_TOKENS = {
    128: 10_000,
    256: 210_000,
    512: 410_000,
    768: 510_000,
    1024: 610_000,
    1536: 710_000,
    2048: 810_000,
    3072: 910_000,
    4096: 1_010_000,
    8192: 1_210_000,
    16384: 1_410_000,
}
WARMUP_CORPUS_START_TOKENS = {
    128: 1_600_000,
    256: 1_650_000,
    512: 1_700_000,
    768: 1_750_000,
    1024: 1_800_000,
    1536: 1_850_000,
    2048: 1_900_000,
    3072: 1_950_000,
    4096: 2_000_000,
    8192: 2_050_000,
    16384: 2_100_000,
}


def _load_v2_generator() -> ModuleType:
    source = Path(__file__).resolve().parent.parent / "prefill-scaling-v2" / "generate.py"
    spec = importlib.util.spec_from_file_location("prefill_scaling_v2_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load shared workload helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    helpers = _load_v2_generator()
    try:
        raw_bytes = CORPUS_PATH.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing local corpus: {CORPUS_PATH}. See workloads/corpora/README.md."
        ) from exc
    actual_corpus_sha256 = _sha256(raw_bytes)
    if actual_corpus_sha256 != CORPUS_SHA256:
        raise RuntimeError(
            "Unexpected WikiText corpus checksum: "
            f"{actual_corpus_sha256}; expected {CORPUS_SHA256}"
        )

    normalized_corpus = helpers._normalize_corpus(raw_bytes.decode("utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.model_max_length = 10_000_000
    corpus_ids = tokenizer.encode(normalized_corpus, add_special_tokens=False)

    rows: list[dict[str, Any]] = []
    sampling: list[dict[str, int]] = []
    for target_tokens in TARGET_TOKENS:
        for profile_role, requested_start_token in (
            ("profile", CORPUS_START_TOKENS[target_tokens]),
            ("warmup", WARMUP_CORPUS_START_TOKENS[target_tokens]),
        ):
            content, source_token_count, sampled_start_token = helpers._exact_window(
                tokenizer,
                corpus_ids,
                requested_start_token,
                target_tokens,
            )
            actual_tokens = helpers._prompt_tokens(tokenizer, content)
            rows.append(
                {
                    "scenario_id": f"{profile_role}-prompt-{target_tokens:05d}",
                    "task_type": "synthetic_prefill",
                    "workload_class": "prefill-heavy",
                    "messages": [{"role": "user", "content": content}],
                    "response_format": "text",
                    "generation": {
                        "temperature": 0,
                        "seed": 42,
                        "max_output_tokens": OUTPUT_TOKENS,
                        "context_window": CONTEXT_WINDOW,
                        "think": False,
                    },
                    "validators": ["exact_match"],
                    "validation": {
                        "exact_text": "OK",
                        "expected_prompt_tokens": target_tokens,
                        "profile_role": profile_role,
                        "corpus_start_token": sampled_start_token,
                        "corpus_source_token_count": source_token_count,
                    },
                    "data_classification": "public",
                    "tags": [
                        "natural-corpus-prefill",
                        "nsight-kernel-scaling",
                        f"profile-role-{profile_role}",
                        "wikitext-2-raw-v1",
                        f"target-prompt-tokens-{target_tokens}",
                        f"fixed-context-{CONTEXT_WINDOW}",
                        "tail-instruction",
                        "fixed-short-output",
                    ],
                }
            )
            sampling.append(
                {
                    "target_prompt_tokens": target_tokens,
                    "profile_role": profile_role,
                    "corpus_start_token": sampled_start_token,
                    "corpus_source_token_count": source_token_count,
                    "actual_constructed_prompt_tokens": actual_tokens,
                }
            )

    scenarios_bytes = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    destination = Path(__file__).with_name("scenarios.jsonl")
    destination.write_bytes(scenarios_bytes)

    manifest = {
        "format_version": "1.0",
        "bundle_id": "synthetic-prefill-kernel-scaling",
        "bundle_version": "1.0.0",
        "created_at": "2026-08-27T00:00:00Z",
        "source": {
            "type": "public_dataset",
            "name": "WikiText-2 Raw",
            "version": "v1",
            "split": "train",
            "url": "https://wikitext.smerity.com/wikitext-2-raw-v1.zip",
            "license": "Creative Commons Attribution-ShareAlike",
            "authors": [
                "Stephen Merity",
                "Caiming Xiong",
                "James Bradbury",
                "Richard Socher",
            ],
            "archive_sha256": CORPUS_ARCHIVE_SHA256,
            "source_file": "wiki.train.raw",
            "source_file_sha256": actual_corpus_sha256,
            "normalized_text_sha256": _sha256(normalized_corpus.encode("utf-8")),
            "normalization": (
                "Removed blank lines and WikiText heading-only lines, joined internal "
                "whitespace, restored WikiText punctuation placeholders, and joined "
                "the remaining prose lines with blank lines."
            ),
        },
        "scenario_count": len(rows),
        "workload_families": ["prefill-heavy"],
        "tokenizer": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "chat_template": "tokenizer-configured Qwen2.5 chat template",
            "normalized_corpus_tokens": len(corpus_ids),
        },
        "sampling": sampling,
        "controlled_dimensions": {
            "expected_prompt_tokens": list(TARGET_TOKENS),
            "context_window": CONTEXT_WINDOW,
            "max_output_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "concurrency": 1,
            "tail_instruction": TAIL_INSTRUCTION.strip(),
        },
        "files": {"scenarios_sha256": _sha256(scenarios_bytes)},
        "intended_use": (
            "Profile attention, MLP, KV, and runtime kernel scaling across the "
            "short-prompt utilization ramp and long-context prefill decline."
        ),
        "limitations": [
            (
                "Each length has one measured natural-language sample and one distinct "
                "shape-warmup sample; clean content variation remains the responsibility "
                "of the benchmark experiment."
            ),
            (
                "Expected prompt counts use the pinned Hugging Face tokenizer; every "
                "profile capture must validate Ollama prompt_eval_count."
            ),
            (
                "The complete source corpus is a required ignored local input and is "
                "not distributed with this repository."
            ),
        ],
    }
    Path(__file__).with_name("manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Corpus tokens after normalization: {len(corpus_ids)}")
    print(f"Wrote {len(rows)} scenarios to {destination}")
    for item in sampling:
        print(
            f"{item['profile_role']}-prompt-{item['target_prompt_tokens']:05d}: "
            f"{item['actual_constructed_prompt_tokens']} prompt tokens, "
            f"source [{item['corpus_start_token']}, "
            f"{item['corpus_start_token'] + item['corpus_source_token_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

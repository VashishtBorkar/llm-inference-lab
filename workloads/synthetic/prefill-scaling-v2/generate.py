from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
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

# Lengths alternate within each pass to reduce correlation with elapsed run time.
TARGET_ORDER = (128, 4096, 512, 16384, 1024, 8192, 256, 2048)

# Widely separated deterministic windows prevent shared natural-language prefixes
# and keep any one article or corpus region from supplying every prompt length.
CORPUS_START_TOKENS = {
    128: 10_000,
    256: 210_000,
    512: 410_000,
    1024: 610_000,
    2048: 810_000,
    4096: 1_010_000,
    8192: 1_210_000,
    16384: 1_410_000,
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _prompt_tokens(tokenizer: Any, content: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return len(input_ids)


def _normalize_corpus(raw_text: str) -> str:
    paragraphs: list[str] = []
    for raw_line in raw_text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = " ".join(raw_line.split())
        if not line or re.fullmatch(r"=+\s.*\s=+", line):
            continue
        line = line.replace(" @-@ ", "-")
        line = line.replace(" @,@ ", ",")
        line = line.replace(" @.@ ", ".")
        line = re.sub(r"\s+([,.;:!?%])", r"\1", line)
        line = re.sub(r"([([])\s+", r"\1", line)
        line = re.sub(r"\s+([])])", r"\1", line)
        line = re.sub(r"\s+'(s|re|ve|m|d|ll|t)\b", r"'\1", line)
        paragraphs.append(line)
    if not paragraphs:
        raise RuntimeError("Corpus normalization produced no text")
    return "\n\n".join(paragraphs)


def _candidate_content(
    tokenizer: Any,
    corpus_ids: list[int],
    start_token: int,
    source_token_count: int,
) -> str:
    passage = tokenizer.decode(
        corpus_ids[start_token : start_token + source_token_count],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not passage:
        raise RuntimeError(f"Corpus window at token {start_token} decoded to empty text")
    return passage + TAIL_INSTRUCTION


def _exact_window(
    tokenizer: Any,
    corpus_ids: list[int],
    start_token: int,
    target_prompt_tokens: int,
) -> tuple[str, int, int]:
    for start_adjustment in range(256):
        candidate_start = start_token + start_adjustment
        if candidate_start + target_prompt_tokens > len(corpus_ids):
            break

        low = 1
        high = target_prompt_tokens
        while low <= high:
            source_token_count = (low + high) // 2
            content = _candidate_content(
                tokenizer, corpus_ids, candidate_start, source_token_count
            )
            actual = _prompt_tokens(tokenizer, content)
            if actual == target_prompt_tokens:
                return content, source_token_count, candidate_start
            if actual < target_prompt_tokens:
                low = source_token_count + 1
            else:
                high = source_token_count - 1

        # Tokenization at the passage/suffix boundary is usually monotonic, but a
        # merge can skip an exact count. Search around the insertion point, then
        # move to the next deterministic corpus boundary if necessary.
        search_start = max(1, low - 32)
        search_stop = min(target_prompt_tokens, low + 32)
        for source_token_count in range(search_start, search_stop + 1):
            content = _candidate_content(
                tokenizer, corpus_ids, candidate_start, source_token_count
            )
            if _prompt_tokens(tokenizer, content) == target_prompt_tokens:
                return content, source_token_count, candidate_start

    raise RuntimeError(
        f"Could not construct an exact {target_prompt_tokens}-token prompt within "
        f"256 corpus tokens of offset {start_token}"
    )


def main() -> int:
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

    normalized_corpus = _normalize_corpus(raw_bytes.decode("utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    tokenizer.model_max_length = 10_000_000
    corpus_ids = tokenizer.encode(normalized_corpus, add_special_tokens=False)

    rows: list[dict[str, object]] = []
    sampling: list[dict[str, int]] = []
    for target_tokens in TARGET_ORDER:
        start_token = CORPUS_START_TOKENS[target_tokens]
        content, source_token_count, sampled_start_token = _exact_window(
            tokenizer,
            corpus_ids,
            start_token,
            target_tokens,
        )
        actual_tokens = _prompt_tokens(tokenizer, content)
        rows.append(
            {
                "scenario_id": f"prompt-{target_tokens:05d}",
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
                    "corpus_start_token": sampled_start_token,
                    "corpus_source_token_count": source_token_count,
                },
                "data_classification": "public",
                "tags": [
                    "natural-corpus-prefill",
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
        "bundle_id": "synthetic-prefill-scaling",
        "bundle_version": "2.0.0",
        "created_at": "2026-08-24T00:00:00Z",
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
            "expected_prompt_tokens": sorted(TARGET_ORDER),
            "context_window": CONTEXT_WINDOW,
            "max_output_tokens": OUTPUT_TOKENS,
            "temperature": 0,
            "concurrency": 1,
            "tail_instruction": TAIL_INSTRUCTION.strip(),
        },
        "files": {"scenarios_sha256": _sha256(scenarios_bytes)},
        "intended_use": (
            "Measure single-request Ollama prefill scaling with deterministic, "
            "non-overlapping natural-language corpus samples."
        ),
        "limitations": [
            (
                "Each prompt length uses one fixed corpus window, so content and "
                "length are not independently randomized."
            ),
            (
                "Expected prompt counts use the pinned Hugging Face tokenizer; "
                "analysis uses Ollama prompt_eval_count."
            ),
            (
                "The complete source corpus is a required ignored local input and "
                "is not distributed with this repository."
            ),
            (
                "Prompt-cache state is a runtime control established by the "
                "experiment launch procedure."
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
            f"prompt-{item['target_prompt_tokens']:05d}: "
            f"{item['actual_constructed_prompt_tokens']} prompt tokens, "
            f"source [{item['corpus_start_token']}, "
            f"{item['corpus_start_token'] + item['corpus_source_token_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

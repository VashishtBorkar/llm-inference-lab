# Workload Design

This document defines how the LLM Inference Lab creates, versions, and validates benchmark workloads. Workloads belong to this repository and should be chosen to answer inference-systems questions rather than mirror any one product.

## 1. Workload Families

Use a small set of complementary workload families.

### Synthetic isolation

Precisely control input tokens, requested output tokens, shared-prefix length, arrival pattern, and request mix. These workloads are best for testing a mechanism such as prefill scaling, decode behavior, continuous batching, or prefix caching.

### Representative tasks

Use purpose-built fixtures or appropriately licensed public data for common serving shapes:

- structured extraction;
- long-document analysis;
- summarization;
- rewriting;
- multi-turn chat;
- code or structured generation where useful;
- mixed short and long requests.

Representative workloads test whether a result survives natural variation in prompts, tokenization, stopping behavior, and output length.

### Stress and boundary cases

Deliberately approach context, memory, concurrency, and arrival-rate limits. OOM, rejection, timeout, preemption, and queue saturation are valid results when captured explicitly.

## 2. Scenario Dimensions

Each workload should declare the dimensions it controls or observes:

- input-token band;
- requested and actual output-token band;
- workload class and expected execution regime;
- shared-prefix token count or ratio;
- request popularity distribution;
- conversation-turn dependencies;
- structured-output constraints;
- concurrency or arrival pattern;
- correctness or quality validators.

Do not use ambiguous labels such as `small` or `large` without recording the actual token counts produced by the selected tokenizer.

## 3. Versioned Bundle Layout

A portable workload bundle should use a self-contained layout such as:

```text
bundle/
  manifest.json
  scenarios.jsonl
  schemas/              # optional response schemas
```

The manifest describes provenance and the dataset as a whole. Each JSONL record describes one engine-neutral request scenario. Engine launch flags and hardware settings belong in the experiment manifest, not the workload.

Once used in a published experiment, a bundle version is immutable. Any content change creates a new version and content hash.

## 4. Bundle Manifest

Record at least:

- `format_version`;
- stable `bundle_id` and immutable `bundle_version`;
- creation timestamp;
- source type: `synthetic`, `public_dataset`, or `private_local`;
- source name, version, URL, license, and transformation notes where applicable;
- tokenizer used to construct token-controlled scenarios;
- scenario count and workload-family summary;
- content hashes for scenario and schema files;
- known limitations and intended experiment use.

Dataset licenses and redistribution constraints must be respected. A public source does not automatically permit committing transformed copies.

## 5. Scenario Record

Each JSONL scenario should include:

- `scenario_id` — stable opaque identifier;
- `task_type` — extraction, summarization, chat, synthetic prefill, or another declared type;
- `workload_class` — prefill-heavy, decode-heavy, interactive, shared-prefix, mixed, or boundary;
- `messages` or `prompt` — fully rendered engine-neutral input;
- `response_format` — text, JSON, JSON Schema, regex, or another declared constraint;
- `response_schema_ref` or inline schema when required;
- `generation` — portable decoding settings and output limit;
- `validators` — deterministic correctness and quality checks;
- `data_classification` — public or private local;
- `tags` — experiment-selection metadata.

Optional fields can define a shared-prefix group, expected input/output token range, request popularity, conversation ordering, or expected failure boundary.

## 6. Illustrative Scenario

This is an example shape, not a finalized JSON Schema:

```json
{
  "scenario_id": "synthetic-extraction-001",
  "task_type": "structured_extraction",
  "workload_class": "prefill_heavy",
  "messages": [
    {
      "role": "system",
      "content": "Extract the requested fields and return valid JSON."
    },
    {
      "role": "user",
      "content": "Synthetic document content is inserted here."
    }
  ],
  "response_format": "json_schema",
  "response_schema_ref": "schemas/extraction-v1.json",
  "generation": {
    "temperature": 0,
    "max_output_tokens": 160,
    "seed": 42
  },
  "validators": ["schema_valid", "required_fields_present"],
  "data_classification": "public",
  "tags": ["long-input", "short-output", "structured"]
}
```

## 7. Portable Generation Settings

Scenarios may specify:

- temperature, top-p, top-k, and seed when supported;
- maximum output tokens;
- context-window size when the experiment must hold KV-cache capacity constant;
- stop sequences;
- repetition or frequency penalties;
- structured-output requirements.

Adapters translate these settings to each engine and must record unsupported, approximated, or changed behavior. Engine-only features are explicit experiment variables; they must not be silently embedded in a workload.

## 8. Token-Controlled Construction

For fixed-token synthetic studies:

- pin the tokenizer revision;
- construct and verify prompts using that tokenizer;
- store the expected token count in the manifest;
- record actual engine-reported token counts;
- avoid relying on repeated whitespace or other content that an engine may normalize;
- distinguish requested maximum output from actual generated output.

When comparing engines, tokenizer or chat-template differences are confounders and must be disclosed.

## 9. Validation

Prefer deterministic, versioned validators:

- JSON parse and schema conformance;
- required keys or values;
- exact-match or bounded-match checks for synthetic tasks;
- truncation and incomplete-output detection;
- language or format constraints;
- stable stopping behavior.

Subjective evaluation is optional. If used, record the evaluator, prompt, rubric, model, and version. Quality outcomes must correspond to the same request records used for performance metrics.

## 10. Privacy and Data Handling

- Synthetic fixtures are the default for source control and public reproduction.
- Public datasets require recorded provenance, license review, and redistribution compliance.
- Private prompts or outputs stay in ignored local storage and are never required to reproduce core results.
- Do not ingest personal application databases, credentials, secrets, usernames, or local file paths.
- Inspect traces and profiler captures before publication because they may embed input or output text.
- Public per-request results should use opaque scenario IDs and omit content unless the scenario is explicitly redistributable.

## 11. Results Linkage

Every experiment result should reference:

- workload format version;
- bundle ID, version, and content hash;
- scenario ID;
- experiment and run ID;
- engine, model, configuration, and environment manifest.

This allows results to remain compact while preserving exact input provenance.

## 12. Evolution

- Backward-compatible optional fields increment the minor format version.
- Breaking field or semantic changes increment the major version.
- The harness declares its supported version range.
- Unknown required fields or unsupported major versions fail before inference begins.
- Published experiments retain the exact bundle and schema hashes used.

Add workload families only when they test a distinct execution behavior or strengthen external validity. A larger dataset is not automatically a better systems benchmark.

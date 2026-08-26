# Local Corpora

External corpus payloads remain local and are ignored by Git. Workload bundles
record source provenance and checksums, while generators verify the local source
before deriving committed scenarios.

## WikiText-2 Raw v1

Experiment 5 uses the raw WikiText-2 training split as natural-language source
material.

- Dataset authors: Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher
- Dataset page: <https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR>
- Archive: <https://wikitext.smerity.com/wikitext-2-raw-v1.zip>
- License: Creative Commons Attribution-ShareAlike
- Archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`
- Required file: `workloads/corpora/wikitext-2-raw-v1/wiki.train.raw`
- Required file SHA-256: `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`

From the repository root:

```bash
mkdir -p workloads/corpora/wikitext-2-raw-v1
curl -fL https://wikitext.smerity.com/wikitext-2-raw-v1.zip \
  -o /tmp/wikitext-2-raw-v1.zip
sha256sum /tmp/wikitext-2-raw-v1.zip
unzip -j /tmp/wikitext-2-raw-v1.zip \
  wikitext-2-raw/wiki.train.raw \
  -d workloads/corpora/wikitext-2-raw-v1
sha256sum workloads/corpora/wikitext-2-raw-v1/wiki.train.raw
```

Do not edit or normalize the source file in place. The versioned workload generator
performs deterministic normalization and sampling.

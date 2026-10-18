# Polyvore feature cache — Eyad's M2 handoff

Status: full accepted-population extraction, assembly and loading verified; final
M2 acceptance is pending. Task: [Build and verify Polyvore feature cache](https://app.notion.com/p/3e098a0a49c081f39b97eaf3039affa0).

On October 18, [source recovery](../data_recovery.md) restored all 14,657
accepted M1 Polyvore validation images. Actual LAB extraction, pinned CLIP inference
and cache assembly now cover this entire population. The earlier five-item run
remains a historical smoke check; it is superseded by the full-population run.

The assembler in `scripts/m2/build_feature_cache.py` consumes Hana's frozen v1.0.0
record shape, the canonical M1 inventory and splits, Eyad's actual LAB palettes, and
Hayat's pinned CLIP extractor. It does not modify those components or choose a final
category model. Polyvore's source categories are mapped using the existing M1
taxonomy; confidence remains null. Structured attributes, season and formality
remain null because validated predictions are unavailable.

## Reproduce

From the repository root, with the project dependencies installed:

```powershell
python -m pip install -e ".[dev,data,ml,color]"
python scripts/tasks.py garments
python scripts/tasks.py colors
python scripts/m2/build_feature_cache.py --extract-embeddings
python scripts/m2/build_feature_cache.py --resume
python scripts/m2/build_feature_cache.py --verify-only
```

The first extraction needs access to the pinned Hugging Face CLIP model, or a local
cache of it. `--batch-size 8` controls CLIP inference batches. After extraction,
assembly/resume and M3 loading checks need no model loading. Every input path can
be overridden; use `--help` for the available options. Recreate palettes after
changing prepared images, and rerun `--extract-embeddings` after changing their bytes.

The full October 18 run used the existing local test runtime and downloaded model
cache. LAB palettes and garment derivatives were prepared first at the explicit
paths below. These overrides are essential: default paths select the older sample.

```powershell
$env:PYTHONPATH='.local-check-tools/model-runtime;.local-check-tools;src'
$env:HF_HOME=(Join-Path (Get-Location) 'artifacts/tester-model-cache')
$env:HF_HUB_OFFLINE='1'
python scripts/m2/build_feature_cache.py --garments artifacts/m1/recovered_population/garment_manifest.csv --palettes artifacts/m2/polyvore_full/v1/color/palettes.jsonl --embedding-dir artifacts/m2/polyvore_full/v1/embeddings --output-dir artifacts/m2/polyvore_full/v1/cache --extract-embeddings --batch-size 32
python scripts/m2/build_feature_cache.py --garments artifacts/m1/recovered_population/garment_manifest.csv --palettes artifacts/m2/polyvore_full/v1/color/palettes.jsonl --embedding-dir artifacts/m2/polyvore_full/v1/embeddings --output-dir artifacts/m2/polyvore_full/v1/cache --resume --batch-size 32
python scripts/m2/build_feature_cache.py --output-dir artifacts/m2/polyvore_full/v1/cache --verify-only
```

Those local runtime directories are ignored by Git. The standard dependency
installation above is the portable setup; the environment overrides only document
the existing local run.

## Outputs and integrity

Generated data stays outside Git:

| Location | Contents |
|---|---|
| `artifacts/m2/polyvore_full/v1/embeddings/` | CLIP vectors, ordered item inventory, metadata, and extraction manifest with image hashes |
| `artifacts/m2/polyvore_full/v1/cache/records.jsonl` | One frozen-contract record per successful item; directly readable by an M3 consumer |
| `artifacts/m2/polyvore_full/v1/cache/inventory.csv` | One disposition per canonical Polyvore item, with split/group metadata and explicit failure or unavailable reason |
| `artifacts/m2/polyvore_full/v1/cache/manifest.json` | Versions, configuration and artifact hashes, counts, loading-check result, and release blockers |
| `artifacts/m2/polyvore_full/v1/cache/entries/` | Atomic item checkpoints containing source traceability and component provenance |

Resume reuses a successful checkpoint only when its inputs, component artifacts,
implementation and image checksum match. Failed entries are retried. Changed
inventories regenerate the final outputs from current items, so old checkpoints
cannot add obsolete or duplicate records. Extraction itself delegates batching to
the existing CLIP implementation; only assembly has per-item restart support.
Run one writer per output directory.

Validation checks JSON Schema plus the existing semantic validator, finite unit
embeddings of dimension 512, palettes and their proportions, source category
mapping, canonical splits and outfit leakage groups. The M3 loading check reads
the serialized JSONL, checks output and input hashes, verifies current image bytes,
and reconciles record IDs against the successful inventory. It verifies the data
interface; it does not evaluate outfit compatibility quality.

## Verified population and remaining gate

The canonical M1 manifest contains **14,657 Polyvore items**. The full October 18
run produced **14,657 successful records, zero failures and zero unavailable
items**, using actual extracted palettes and normalized 512-dimensional CLIP
vectors. Every embedding input has a recorded image checksum. Both the built-in
loading check and a separate `--verify-only` passed for all 14,657 records. Earlier
five-item runs and regression tests verified resume; the full-population run was
fresh extraction/assembly followed by independent readback verification.

This is complete coverage of the accepted M1 Polyvore validation population.
Newly recovered train/test populations are outside that frozen inventory and have
not yet been embedded or integrated for M3. Source categories retain null confidence;
attributes remain unavailable because no trained predictor is supplied.

Final release requires the team to settle component and taxonomy approval in
[model selection](model_selection_decision.md), review the handoff, and approve an
evaluation policy for the recovered cross-split overlap. See the
[readiness audit](../audit/M1_M2_READINESS_2026-10-18.md). `handoff_ready` stays false;
a passing command does not approve a milestone. Hana's contract shape is usable,
but her Notion Completed status conflicts with its unchecked Handoff Ready field.

## Checks

```powershell
python -m pytest tests/test_feature_cache.py
python scripts/tasks.py lint
python scripts/tasks.py format-check
python scripts/tasks.py typecheck
python -m pytest
```

The cache tests cover assembly and loading, repeat/resume, stale images and inputs,
model revision and vector validation, duplicate IDs, split leakage, mismatched
component splits, invalid palettes, corrupt outputs, missing embeddings and output
path protection. Tests use explicitly synthetic fixtures; the full-population run above
uses real extracted features.

October 18 validation: **138 tests and six subtests passed** after the audit fixes.
Lint, formatting, package type checking, repository validation, artifact policy and
documentation links passed. Tests used an explicit workspace `--basetemp` because
the sandbox's default pytest temporary directory was inaccessible. The local
runtime paths under `.local-check-tools` supplied test dependencies and type stubs.
Full extraction and loading evidence is recorded under
`artifacts/m1-m2-audit-20261018/` and the full cache manifest.

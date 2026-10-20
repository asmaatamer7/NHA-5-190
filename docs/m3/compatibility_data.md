# M3 compatibility data and Hayat's feature handoff

Eyad owns the data examples and final cache checks. Hayat leads CLIP extraction
and shared compatibility features. [Notion task](https://app.notion.com/p/3dd98a0a49c08183b6f2cb9b7e50e96a).

## Three commands, three stages

Work directly in the repository after pulling the M3 changes. No ZIP, code-copy
step or separate helper package is required. The implementation, tests and this
guide live in `scripts/m3/`, `tests/` and `docs/m3/` respectively.

Generated inventories, images and features are local artifacts, not Git files.
The `pairs` command below regenerates the exact feature request from the recovered
Polyvore source files. It needs all three source splits and their metadata; the
later `training-images` stage needs only that request and the training parquet.
On a machine without the M2 validation cache, data preparation still reports its
missing coverage; obtain the verified cache before claiming integration complete.

The implementation is one CLI: `scripts/m3/prepare_compatibility_data.py`.
It never changes original sources, M1 manifests, taxonomy or the existing M2 cache.
Use new output directories on reruns; an existing version is not overwritten.

```powershell
python -m pip install -e ".[data,ml,color]"
# Inventory and split audit only:
python scripts/m3/prepare_compatibility_data.py --stage audit --output artifacts/m3/compatibility_data/audit_v3
# Complete data examples and the exact training feature request:
python scripts/m3/prepare_compatibility_data.py --stage pairs --output artifacts/m3/compatibility_data/handoff_v1
```

`python scripts/tasks.py m3-data` defaults to the audit stage. For another machine,
use `--source-root` to locate the recovered Polyvore source directory. Defaults
otherwise resolve against the repository, independently of the shell directory.

## Data protocol

- Preserve test. Filter validation against test by item ID and exact image bytes.
  Filter training against both original held-out populations, including removed
  validation items. Original assignments remain traceable; items are not moved.
- Remove whole affected outfits, compatibility rows and FITB questions, including
  questions with an ineligible candidate answer. Never edit individual answers.
- Check shared item IDs, exact image hashes and source outfit IDs. This does not
  prove freedom from near duplicates or alternate photographs of one product.
- Keep unknown common categories empty, with `mapping_status=unmapped`; preserve
  the original source category ID. Check every source ID against official
  `categories.csv`. No common-category mapping is added or guessed. See the
  [Notion mapping policy](https://app.notion.com/p/3ea98a0a49c08154a47bcd5b9cd7ae78).
- Positive pairs represent source-outfit co-occurrence, not human pair labels.
  Use different source categories and image hashes. Deduplicate pairs/self-pairs.
- Negative replacements match the replaced item's exact original category ID,
  stay in the same split, come from another outfit and cannot match any known
  positive ID pair or exact-image-hash pair in that split. Test positive labels
  never inform training negatives. Negatives are explicitly assumed; noise remains.
- Use seed 42, sorted inputs and LF JSONL exports. Try at most 256 replacement
  candidates per positive. Count missing negatives rather than inventing them.

Relevant category pairs for mean OCS and valid candidate-outfit structures remain
Hana/Omar's separate decisions. Data preparation does not approve an evaluation
protocol, close M2/M3 or train a compatibility model.

## Outputs

| Output | Contents |
|---|---|
| `item_inventory.csv` | Every original item, split eligibility, IDs, source category, common mapping and original image hash |
| `excluded_items.jsonl`, `excluded_examples.jsonl` | Explicit exclusions with source references and reasons |
| `<split>/outfits.json`, `compatibility.txt`, `fitb.json` | Retained official examples with original references/answers |
| `<split>/pairs.jsonl` | Deterministic positive and assumed-negative pairs with IDs and label provenance |
| `missing_feature_items.jsonl` | Referenced items absent from the current M2 cache |
| `feature_request/` | Exact train/validation item inventories and checksummed training request |
| `report.json` | Counts, input/output/code hashes, limitations and readiness flags |

Partial outputs without a final report are not a successful build. Readiness stays
false until actual feature coverage and the required reviews are complete.

## Hayat: prepare only the requested training images

The training request needs the official training parquet only. It does not need
training/test images to be redistributed through Git. Obtain
`data/disjoint/train.parquet` from `mvasil/polyvore-outfits`, dataset revision
`22cc0c748c293a598e1df561a76abe96273cb76a`, using approved access. Place it under
`data/raw/polyvore_outfits/`. The request verifies its SHA-256 before reading it.

```powershell
python scripts/m3/prepare_compatibility_data.py --stage training-images --request artifacts/m3/compatibility_data/handoff_v1/feature_request/training_request.json --output artifacts/m3/training_inputs/v1
```

This verifies the request CSV, exact required IDs and original image hashes,
extracts only requested training photos, and reuses Hana's existing
`full_image_source` preparation. It writes `garment_manifest.csv` and
`preparation_report.json`. Both original-photo and prepared-photo checksums are
preserved. No CLIP extraction occurs in this stage. Test items are rejected.

## Hayat: run the existing pinned CLIP extractor

Run from the repository root after image preparation. This calls the existing
extractor directly, because its old CLI has a fixed sample input path.

```powershell
python -c "import torch; torch.set_num_threads(8); from wardiq.representation.clip_embeddings import extract_clip_embeddings; extract_clip_embeddings('artifacts/m3/training_inputs/v1/garment_manifest.csv', 'artifacts/m3/training_embeddings/v1', image_root='.', batch_size=32)"
```

Model: `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; float32, 512 dimensions, L2 normalized.
The existing extractor chooses CUDA when available, otherwise CPU. The first run
needs the pinned weights or a matching local cache. Existing extraction validation
checks vector shape, finite values, unit norms and inventory alignment. No model
architecture or training code is changed. CLIP extraction is not resumable; after
an interruption, retain prepared images and restart extraction into a new output.

For LAB features, reuse Eyad's existing extractor on the same prepared manifest:

```powershell
python scripts/extract_color_palettes.py --manifest artifacts/m3/training_inputs/v1/garment_manifest.csv --output-dir artifacts/m3/training_colors/v1
```

Record the full training scope in the handoff; the existing color configuration's
sample-scope label does not describe this larger run. Do not change its algorithm
parameters or source component version without recording a new configuration.

## What Hayat returns

- `clip_embeddings.npy`, ordered `clip_inventory.csv` and `metadata.json`.
- `garment_manifest.csv` and `preparation_report.json` with image checksums.
- If running LAB too: `palettes.jsonl` and `coverage.json`, with actual counts.
- Actual extraction environment/device, command, errors and limitations.

Keep generated data and weights outside Git. Preserve original manifests and
metadata for traceability; do not hand-edit IDs, row order or hashes. Eyad will
verify returned IDs/model revision/dimensions/checksums and assemble the final
cache. Unmapped common categories and unavailable attributes remain unknown.
The final M2 cache assembler must explicitly support these documented unknowns;
its current strict handling of missing taxonomy entries is not silently bypassed.

## Debugging and tests

```powershell
python -m pytest tests/test_m3_compatibility_data.py
python scripts/tasks.py lint
python scripts/tasks.py format-check
```

Errors identify the affected ID or input and return a failing exit status. Fix
source placement or regenerate a changed request; never disable checksum checks.
Tests cover split/label integrity, repeatability, LF serialization, whole FITB
filtering, repeated item positions, source preservation, output protection,
training-only image requests and compatibility with Hayat's manifest loader.

## Current full-data execution

The working split filter retains 68,075 training, 14,615 validation and all 70,035
test items. There are 11,266 / 2,946 / 15,145 retained source outfits respectively.
The generated pair populations require 50,567 training item features and 14,416
validation item features. All 14,416 required validation IDs are in the verified
M2 cache; training extraction remains Hayat's step. The 81 training / 109 test
items outside the common-category lookup are retained as explicitly unmapped.
Test-source examples are preserved for later locked final evaluation, not tuning.

# Hayat's M3 work: masked pair features and training extraction

Prepared by Eyad with Codex to help with Hayat's assigned M3 work. The original
repository files are unchanged. This ZIP contains new files for review; it does
not apply changes, commit, push, approve a protocol or close M2/M3 automatically.

## Use the supplied results first

Read `verification/delivery_report.json` for the actual completed populations and
checks. A full-run result is complete only when its final report exists and its
file hashes pass. Partial chunks/logs do not establish completion.

The delivery contains the new source files under `src/`, `scripts/m3/`, `tests/`
and `configs/m3/`. Generated results are under `results/`. No raw photos, model
weights or held-out test pair labels are included. Local paths in provenance
describe the producing machine; rerunning preparation generates fresh paths.

The scripts can run from this extracted directory with the existing WARDIQ
repository installed. Integration later requires adding only these new files to
the matching repository paths. Do not overwrite existing files automatically.
The exact producing repository commit is recorded in the delivery report.

From the existing WARDIQ repository root:

```powershell
python -m pip install -e ".[dev,data,ml,color]"
# Set this to the directory where you extracted this ZIP:
$delivery = "artifacts/m3/hayat_delivery"
python -m pytest "$delivery/tests"
```

To verify every delivered file without loading large arrays into RAM, run this
Python snippet after setting `root` to the extracted directory:

```python
import hashlib
import json
from pathlib import Path

root = Path("artifacts/m3/hayat_delivery")
for name, expected in json.loads((root / "checksums.json").read_text()).items():
    digest = hashlib.sha256()
    with (root / name).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == expected, name
print("All delivered file checksums passed")
```

## Feature definition

The feature schema is `wardiq.pair-features.v1`. Its explicit column names and
configuration fingerprint must accompany every matrix and model. The supplied
configuration has **1,214 float32 columns**, in this exact order:

| Block | Columns | Meaning |
|---|---:|---|
| Category pair | 78 | Unordered one-hot pair of 11 existing common categories plus unknown |
| Category mask | 1 | Both common categories available |
| CLIP | 1,025 | Cosine, 512 absolute differences, 512 elementwise products |
| LAB | 5 | Mean-color Delta E76, palette-weighted expected Delta E76, lightness difference, chroma difference, both-colors-available mask |
| Attributes | 105 | For each of 35 existing provisional names: absolute difference, product, both-observed mask |

Category pairs follow `category_pairs` in the config; attributes use alphabetical
names, independently of the original attribute-ID vector order. Attribute joins
are by name. The feature schema uses only the existing 35-name provisional
vocabulary; it does not establish that these are trained/supported labels.

LAB Delta E76 distances and lightness differences are divided by 100; chroma
differences by 181. They are numerical descriptors, not calibrated compatibility
scores or probabilities. Mean colors and expected distances use palette area
proportions. Background contamination remains a limitation of product-photo LAB
extraction. No new subjective color-harmony rule is introduced.

Pair order is canonicalized by item ID. Swapping the two items gives byte-exact
identical features. No labels, population statistics, thresholds, validation/test
fitting, normalization fitting or model training enter this transformation.

Missing attributes produce zero placeholders **with observation mask zero**.
A genuinely observed zero score has observation mask one. Missing color is also
explicitly masked. Missing/invalid CLIP embeddings are rejected, not fabricated.
Polyvore has no supported attribute scores in these inputs: all 35 attribute
blocks remain masked. Do not describe their placeholders as negative labels.

Unknown common categories stay unknown. No source-category mappings were added.
The new training component view (`wardiq.pair-feature-item.v1`) stores every
source outfit reference and original/prepared image hashes. It is an internal
feature input, **not an approved M2 cache** or a change to the frozen M2 schema.
Outfit references in this view cover the supplied retained training outfits;
the original pair records keep their separate label/replacement provenance.
The builder also accepts existing frozen-schema v1.0.0 M2 records.

## Read the matrices

```python
import json
from pathlib import Path
import numpy as np

root = Path("artifacts/m3/hayat_delivery/results/features")
X_train = np.load(root / "train/features.npy", mmap_mode="r", allow_pickle=False)
y_train = np.load(root / "train/labels.npy", allow_pickle=False)
X_validation = np.load(root / "validation/features.npy", mmap_mode="r", allow_pickle=False)
y_validation = np.load(root / "validation/labels.npy", allow_pickle=False)
schema = json.loads((root / "train/feature_schema.json").read_text())
assert X_train.shape[1] == X_validation.shape[1] == schema["dimension"]
```

`pair_inventory.csv` binds each row to original item IDs, split, label and label
origin. Never shuffle it separately from features/labels. Source-outfit
co-occurrence positives and category-matched replacement negatives are weak
supervision; replacement negatives are assumed, not human judgments.

Ziad can fit any scaling/constant-column removal on **training only**, then reuse
that fitted transformation for validation/scoring. The 105 masked attribute
columns are constant for the current Polyvore inputs. Model training and the
rule baseline remain the respective owners' tasks.

## Reproduce extraction without changing existing code

Use the existing data-preparation CLI on the canonical request. The request and
CSV must stay together. Original dataset access and pinned model weights are
required for a fresh run; neither is distributed in this ZIP.

```powershell
python scripts/m3/prepare_compatibility_data.py --stage training-images --request "$delivery/inputs/feature_request/training_request.json" --output artifacts/m3/hayat_rerun/training_inputs
python "$delivery/scripts/m3/extract_training_features.py" --manifest artifacts/m3/hayat_rerun/training_inputs/garment_manifest.csv --output artifacts/m3/hayat_rerun/training_embeddings --chunk-size 2048 --batch-size 32 --threads 6
python "$delivery/scripts/m3/extract_training_colors.py" --manifest artifacts/m3/hayat_rerun/training_inputs/garment_manifest.csv --config "$delivery/configs/m3/training_colors.yaml" --output artifacts/m3/hayat_rerun/training_colors --workers 6 --chunk-size 2048
```

The new wrapper calls the existing pinned CLIP extractor unchanged. It checkpoints
chunks, validates model revision, item order, dimensions, norms and file hashes,
then merges the results. Restart an interrupted run with the same command plus
`--resume`. Completed chunks are revalidated; a corrupt completed chunk fails
rather than silently being trusted. Changed inputs/settings require a new run
directory. Prepared photos are hash-verified before and after extraction.

Model: `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, 512-dimensional L2-normalized float32.
The existing extractor uses CUDA if available, otherwise CPU. Our run used CPU.
The LAB configuration preserves existing algorithm parameters and records the
full requested training scope; it is not a new color algorithm.
The LAB wrapper launches the existing color CLI on disjoint chunks, using one
numerical thread per worker, and merges the verified records in original item
order. It validates counts, IDs, source/configuration hashes and completion
receipts. Its `--resume` option reuses intact completed chunks, including when
the worker count changes. It does not reuse changed chunks or inputs. The
parallel wrapper passed a real eight-image check before the full run.

## Reproduce the component join and pair matrices

For a feature-only rebuild from the completed results ZIP, no photos or model
inference are needed. In the first command below, use the delivered
`$delivery/results/training_inputs/garment_manifest.csv`,
`$delivery/results/training_embeddings` and
`$delivery/results/training_colors/palettes.jsonl` as the prepared/embedding/color
inputs. Their checksums remain valid even when original photo paths refer to the
producing machine. The commands shown otherwise describe a complete fresh rerun.

```powershell
python "$delivery/scripts/m3/assemble_feature_items.py" --request "$delivery/inputs/feature_request/training_request.json" --prepared artifacts/m3/hayat_rerun/training_inputs/garment_manifest.csv --embeddings artifacts/m3/hayat_rerun/training_embeddings --colors artifacts/m3/hayat_rerun/training_colors/palettes.jsonl --outfits "$delivery/inputs/train/outfits.json" --output artifacts/m3/hayat_rerun/training_feature_items
python "$delivery/scripts/m3/build_pair_features.py" --records artifacts/m3/hayat_rerun/training_feature_items/records.jsonl --pairs "$delivery/inputs/train/pairs.jsonl" --config "$delivery/configs/m3/pair_features_v1.json" --output artifacts/m3/hayat_rerun/features/train --split train
python "$delivery/scripts/m3/build_pair_features.py" --records "$delivery/inputs/validation_cache/records.jsonl" --pairs "$delivery/inputs/validation/pairs.jsonl" --config "$delivery/configs/m3/pair_features_v1.json" --output artifacts/m3/hayat_rerun/features/validation --split validation
```

Every requested item must resolve. Duplicate IDs, invalid labels, cross-split
rows, duplicate unordered pairs, self pairs, model/version mismatches, invalid
palettes/vectors and inconsistent masks fail with an explicit error. New outputs
never overwrite an existing version. Reports include zero unresolved references,
input/output hashes, dimensions and configuration identity only after success.

For scoring after integration, use `PairFeatureBuilder(config, split)` and
`transform_pair(left_id, right_id)`. This is the same function used for training
exports. For new wardrobe records with null target_split, use split `inference`.
This builder generates features; a separately trained model generates scores.

## Debugging and review

- Missing source: obtain the canonical Polyvore training parquet and request;
  do not substitute another dataset revision.
- Checksum mismatch: preserve the failing ID/path and recover the original input.
- RAM pressure: load feature matrices with memory mapping and lower CLIP batch
  size. Changing batch/chunk settings requires a new extraction directory.
- An existing output without its final report is incomplete. Do not use its
  partial matrix. Preserve logs and select a new output version.
- Do not disable masks, provenance or split checks to make a run pass.

Final protocol acceptance, missing M2 model approvals and final cache integration
are still team review gates. The source split/hash filter does not establish
freedom from near-duplicate/product-level leakage. No test-set feature generation,
compatibility training, scoring/ranking evaluation or milestone closure is claimed.

Task: https://www.notion.so/3fd98a0a49c08154bb1fc4e40034a6ee

The shipped runner also refuses `--resume` in an unrelated directory without its
original `run.json`. The full extraction used the producer snapshot retained in
`verification/extraction_runner_used.py`; `runner_revision.json` binds its hash to
the extraction metadata and documents this later entry-check guard. The guarded
runner passed a separate eight-real-image smoke run and the final focused tests.

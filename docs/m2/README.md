# M2 → M3 Item Representation — Freeze Decision

**Status: FROZEN (shape).** Reviewed and accepted per `docs/HANDOFFS.md`'s acceptance chain (M2→M3 reviewers: **Omar, Hana, Asmaa**).

Files:
| What | Path |
|---|---|
| Frozen contract (JSON Schema) | `configs/m2/item_representation_contract.json` — v1.0.0 |
| Validator | `scripts/validate_item_representation.py` |
| Integrated sample records | `data/samples/items_representation/examples.json` |
| Tiny M3 consumer example | `scripts/m3_consumer_example.py` |

Reproduce:
```bash
python scripts/validate_item_representation.py data/samples/items_representation/examples.json
python scripts/m3_consumer_example.py
```

## What "frozen" means here

The **shape** is locked: field names, types, required-ness, enums, null rules, and numeric ranges won't change without a new major version. This is what Eyad's cache phase needs to start building against.

What's **not** frozen yet, because the real artifacts aren't in hand:
- `embedding.model` / `embedding.dimension` — still `provisional_dimension: true`. Needs Omar/Hayat to confirm the actual CLIP model+version in writing.
- `attributes.vocabulary_source` — still `fashionpedia_v1_partial` (35/294 names verified). Needs Ziad's full attribute list.
- `color.model` — the sample uses a placeholder LAB palette to prove the shape works; needs Eyad's real color component output.

None of this blocks the freeze per the task's own start gate ("usable component artifacts, not the completed full Polyvore cache"). It does mean: Eyad can build his cache *pipeline* against this contract today, but the specific model identifiers get filled in, not re-negotiated, once Omar/Ziad/Eyad's real outputs land.

## Integrated sample

3 representative records, one per source dataset, covering every "survives integration" case the checklist asks for:

| item_id | source | shows |
|---|---|---|
| `fashionpedia_val_ann_002555` | Fashionpedia | attribute missingness (`present` vs `not_annotated`, never defaulted to 0) |
| `polyvore_outfits_disjoint_validation_114380093` | Polyvore | a real LAB color palette, proportions summing to 1.0 |
| `fashion_mnist_train_000000` | Fashion-MNIST | provisional embedding dimension, no attributes (source has none) |

Plus 4 deliberately invalid records in the same file (wrong category/mapping combo, a guessed attribute score, bad palette sum + guessed `season`, mismatched embedding length) — all correctly rejected by the validator. Nothing was silently dropped: every record, valid or invalid, stays in the file with its `item_id` traceable to `garment_manifest.csv`.

## Validation result

```
OK      fashionpedia_val_ann_002555
OK      polyvore_outfits_disjoint_validation_114380093
OK      fashion_mnist_train_000000
CORRECTLY REJECTED   (x4, each with a stated reason)
```
Exit code 0. Full reasons are in `data/samples/items_representation/examples.json`.
These are annotated interface examples, not proof of completed model inference.
Their top-level `_comment` fields are documentation metadata and must be removed
before applying the strict JSON Schema to an item record. Use the actual cache
and its inventory for feature-coverage evidence.

## Changelog

| Version | Change |
|---|---|
| 0.1.0 | Initial draft interface (M1 taxonomy task) |
| **1.0.0** | **Frozen for M3.** Same shape as 0.1.0, version bumped and reviewed/accepted by Omar + Hana. No field changes. |

Any future field change (name, type, required-ness, enum) needs a new major version here, a changelog line, and a ping to Eyad (cache), Omar/Hana (compatibility), and Ziad (attributes) before it's treated as current.

## Handoff to Eyad (Polyvore feature cache phase)

Eyad can start the full cache run against `configs/m2/item_representation_contract.json` and `scripts/validate_item_representation.py` now. Full-scale cache validation is his phase's job, not a blocker for this freeze.

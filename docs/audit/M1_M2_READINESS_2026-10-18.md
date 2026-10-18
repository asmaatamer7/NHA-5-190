# M1/M2 readiness audit — October 18, 2026

**Verdict: the tested software and recovered M1 data work, but M1/M2 cannot yet
be certified as fully accepted for the handoff into M3.** The remaining gates
are concrete data/model decisions and unfinished model deliverables, not failing
infrastructure tests. M3 has not started; the split finding below concerns the
validity of future training/evaluation. No Notion milestone was marked complete
by this audit.

## Verified execution

| Check | Result |
|---|---|
| Final pytest suite after fixes | 138 tests and six subtests passed |
| CI-scope lint, formatting and package type checks | Passed |
| Repository structure, artifact policy and documentation links | Passed |
| Python compilation across package, scripts, tests and M2 configuration/report generators | Passed |
| Cleaning, canonical splits, taxonomy coverage, duplicate references, sample images, EDA and attribute-target handoff | All commands passed |
| Recovered raw-manifest regeneration | Exact match to the preserved 85,815-row M1 inventory |
| Full prepared-population paths | 23,443 accounted garment rows; 19,350 usable derivatives; zero missing usable paths |
| Image/garment loaders on recovered data | Required fields and finite 3×224×224 tensors passed across sources; 37 garment examples plus representative image-level examples |
| M3 consumer example | Runs successfully from another working directory |
| Local synthetic ResNet checkpoint replay | Four of four saved predictions reproduced; this is pipeline evidence, not garment-model quality |
| Full accepted M1 Polyvore cache | 14,657 successful records, zero failed/unavailable; built-in loading check and separate verify-only both passed |

Full raw data integrity checks cover native Fashion-MNIST image/label structures
and checksums, every Fashionpedia validation image, and every downloaded Polyvore
train/validation/test image. Fashionpedia's recovered training ZIP passes CRC and
contains all 45,623 annotated training photos. Its training annotations contain
333,401 instances and 294 attribute names. All recovered Polyvore compatibility
and FITB references resolve; their labels and correct-answer references pass
structural checks. These checks do not fit a model or evaluate test predictions.

## Fixes made

- Corrected the M3 consumer's sample path and made it independent of the calling
  directory. Corrected its documented reproduction paths.
- Made the public item validator enforce the frozen JSON Schema, reject
  nonfinite/non-JSON values, and require null scores for unavailable attributes.
  Annotated example envelopes explicitly strip `_comment`; actual item records
  reject undocumented fields. The validator now requires the declared data extra.
- ResNet checkpoint loading constructs the architecture without downloading
  ImageNet weights; training retains its existing pretrained initialization.
  Reload validation preserves numeric-looking IDs/labels as strings and fails on
  prediction mismatch. Model architecture, training method and existing metrics
  were not replaced.

## Blocking handoff findings

| Gate | Evidence and next action | Owner/reviewer |
|---|---|---|
| Polyvore evaluation leakage | Source-labelled disjoint files share 3,781 train/validation IDs, 84 train/test IDs, and 34 validation/test IDs. There are 3,899 exact image hashes appearing in multiple splits. Review filtering/grouping before training or reporting M3 evaluation. | Omar with downstream evaluation reviewers |
| Supported attribute model | Existing targets/masks and their tests work. The submitted inference configuration still has no trained checkpoint, empirical thresholds or validation precision/recall/F1. Newly recovered training data has not been integrated into this component. | Ziad; Hana/Asmaa review |
| Final category/embedding decision | The decision record retains CLIP as provisional, and Fashion-MNIST ResNet as a pipeline baseline. It does not select an accepted real-photo category predictor. A source-category-only M3 handoff or deferral must be explicit. | Omar; Hana/Asmaa review |
| Acceptance evidence | Taxonomy still says pending lead review; M1/M2 Notion milestones are In Progress. Several tasks say Completed while Handoff Ready is unchecked and evidence fields remain placeholders. Resolve these against actual artifacts and reviewer responses. | Project lead and named handoff reviewers |

The current JSON record shape is frozen at v1.0.0, but a frozen shape does not
approve model choices. Source categories with null confidence are annotations,
not classifier predictions. Polyvore's null attribute scores are honest source
missingness, not evidence of a completed attribute predictor. Color provenance
retains the unmasked product-photo background risk.

## Concrete split proposal for review

An **unapproved analysis only** preserves the entire source test set, removes
validation items matching test by ID or exact image bytes, and removes training
items matching either held-out population. Whole compatibility/FITB examples are
excluded if any reference is ineligible; candidate answers are not silently
deleted. It produces zero cross-split ID or exact-byte-hash overlap:

| Split | Eligible items | Eligible outfits | Compatibility examples | FITB questions |
|---|---:|---:|---:|---:|
| Train | 68,075 | 11,266 | 22,380 | 8,783 |
| Validation | 14,615 | 2,946 | 5,892 | 2,914 |
| Test | 70,035 | 15,145 | 30,290 | 15,145 |

No official files or canonical assignments were changed. This is a reviewable
proposal, not a claim that the team approved that protocol. Exact-byte checks do
not establish freedom from near-duplicate or product-level aliases.

## Feature-cache execution and evidence

LAB extraction and pinned CLIP inference completed on all **14,657 accepted M1
Polyvore items**, with zero failures or exclusions. Strict assembly produced
**14,657 successful records, zero failed and zero unavailable** under
`artifacts/m2/polyvore_full/v1/`. The built-in loading check and a separate
`--verify-only` both passed, reconciling all serialized records, inventory,
checkpoints and current source-image checksums.

Embeddings are float32 with shape 14,657×512, all finite, with L2 norms between
0.99999982 and 1.00000012. All records retain source categories with null confidence
and null unavailable attribute scores. The cache remains explicitly provisional,
with `handoff_ready: false`. This is the full accepted validation population, not
an assertion that every newly recovered source item is integrated.

Evidence: `full_cache_execution.json`, `full_cache_verify.json`,
`full_cache_statistics.json` in the audit artifact directory, plus the cache
`manifest.json`. Earlier five-item resume checks and regression tests cover
checkpoint reuse; this full-population run was fresh extraction/assembly followed
by independent readback verification, not a full-population resume run.

The accepted M1 cache population is validation-only. The additionally recovered
train/test populations remain outside that frozen inventory and still need an
approved M3 eligibility/cache integration plan. They must not be treated as
already integrated simply because their source files are present.

Detailed logs and machine-readable results are under
`artifacts/m1-m2-audit-20261018/`, including quality/pipeline results, loader checks,
official split and reference audits, the complete overlap CSV, proposed
eligibility, checkpoint replay, and final test output. Generated datasets,
features and reports remain ignored by Git.

The actual Fashion-MNIST baseline/fine-tuned checkpoints were not available
locally for fresh replay; their existing run reports remain separate evidence.
No real-garment training, new empirical attribute metrics, or final model quality
was invented by this audit.

## Source requirements

- [Notion M1](https://app.notion.com/p/3dd98a0a49c081abae84f59e3e4a21bc)
- [Notion M2](https://app.notion.com/p/3dd98a0a49c081bb84f1c46325fb0421)
- [Cache acceptance](https://app.notion.com/p/3e098a0a49c081f39b97eaf3039affa0)
- [Final representation handoff](https://app.notion.com/p/3dd98a0a49c081a68d63f902ddd33290)
- [Repository handoff acceptance chain](../HANDOFFS.md)
- [Model decision record](../m2/model_selection_decision.md)

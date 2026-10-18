# Dataset recovery — October 18, 2026

The original accepted M1 source files have been restored under `data/raw/`.
Rebuilding the raw manifest produced **85,815 rows** and matched the preserved
repository manifest exactly after LF normalization:

```powershell
python scripts/build_raw_manifest.py
python scripts/compare_manifest.py
```

| Source | Verified population | Local location |
|---|---|---|
| Fashion-MNIST | 60,000 train and 10,000 test images/labels; official gzip MD5 checks pass | `data/raw/fashion_mnist/`, compressed and extracted IDX files |
| Fashionpedia validation | 1,158 images, 8,781 annotations, 46 categories, 294 attributes; archive CRC, image decoding, dimensions and canonical IDs verified | `data/raw/fashionpedia/val_test2020.zip`, `instances_attributes_val2020.json`, extracted `validation/` photos |
| Polyvore disjoint validation | 14,657 unique images; all decode and all canonical IDs match | `data/raw/polyvore_outfits/disjoint/validation-00000-of-00001.parquet`, extracted `images/` photos |
| Polyvore metadata | 251,008 entries; published SHA-256 verified; validation outfit JSON restored | `data/raw/polyvore_outfits/polyvore_item_metadata.json`, compatibility copy `metadata.json`, and `disjoint/valid.json` |

The Polyvore parquet was recovered from `C:/Users/Hp/Downloads/0000.parquet`.
Its SHA-256 matches the historical inspection report:
`5fdd4413b9fc0fc42d21e2215bdb04a97301b4fc1852b32c155cdeadd98ac331`.
Embedded JPEGs were extracted without recompression.

Fashionpedia came from the [official download host](https://github.com/cvdfoundation/fashionpedia).
Fashion-MNIST was restored with the existing repository downloader.
Polyvore auxiliary files came from its [official dataset repository](https://huggingface.co/datasets/mvasil/polyvore-outfits),
pinned to revision `22cc0c748c293a598e1df561a76abe96273cb76a`.
Downloads use the local Hugging Face login; credentials are absent from project
configuration and reports.

## Additional training data

Fashionpedia's official training annotations and training-image archive are
recovered: 45,623 image records, 333,401 annotations, 46 categories and 294
attributes. The training ZIP passes CRC validation and contains every annotated
image. Polyvore's source-labelled disjoint training/test parquets and outfit,
compatibility and FITB files are also recovered. They contain 71,967 training and
70,035 test item records; all images decode and all outfit references resolve.

**M3 evaluation blocker:** the downloaded Polyvore splits share item IDs:
train/validation 3,781; train/test 84; validation/test 34. There are also 3,899
image-byte hashes present in multiple splits. Do not infer leakage freedom from
the source folder name. The complete ID overlap inventory and checks are in
`artifacts/m1-m2-audit-20261018/`; the split owner must review a policy before
these populations are used for learned compatibility evaluation.

These additional populations remain separate from the accepted M1 inventory.
Recovery does not change frozen splits, train models, or establish model quality.
Test data must remain isolated from model selection.

## Prepared inputs and evidence

The ignored `artifacts/data_recovery/` directory contains source URLs/revisions,
byte counts, hashes, per-image inventories and validation summaries.
The generated preparation configuration uses the existing garment preparer with
the recovered validation populations and canonical split assignments:

```powershell
python scripts/prepare_garment_crops.py --config artifacts/data_recovery/preparation/garment_preparation.yaml
```

Its outputs are separate from the sample outputs:
`artifacts/m1/recovered_population/` and `data/processed/recovered_garment_crops/`.
Preparation completed with 23,443 accounted garment rows: 14,657 Polyvore product
photos, 4,688 Fashionpedia garment crops, 4,093 explicitly skipped parts/decorations,
and five Fashion-MNIST smoke-test previews. The image-level manifest has 15,820
source images. Polyvore keys exactly match the accepted inventory, every prepared
Polyvore path resolves, and canonical split assignments are preserved.
The default preparation command still targets committed samples. Subsequent full
color, embedding and cache runs must explicitly use the recovered-population
manifest. The earlier five-item cache remains sample evidence until full execution
and review take place.

Raw datasets, derived images and generated reports are ignored by Git and must
stay outside source-code commits.

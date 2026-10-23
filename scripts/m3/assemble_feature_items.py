"""Join verified training components into a feature view, not an accepted M2 cache.

The separate internal view preserves all outfit references instead of inventing
a single frozen-contract leakage_group_id for items appearing in several outfits.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from extract_training_features import digest, write_json


def keyed_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keyed = {r["item_id"]: r for r in rows}
    if len(keyed) != len(rows) or "" in keyed:
        raise ValueError(f"Missing or duplicate inventory IDs: {path}")
    return keyed


def assemble(request_path, prepared, embeddings, colors_path, outfits_path, output):
    from wardiq.representation.clip_embeddings import CLIP_MODEL_ID, CLIP_MODEL_REVISION

    request_path, prepared, embeddings, colors_path, outfits_path, output = map(
        Path, (request_path, prepared, embeddings, colors_path, outfits_path, output)
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("version") != "wardiq.m3.training-feature-request.v1":
        raise ValueError("Unsupported training feature request")
    if (
        request.get("model_id") != CLIP_MODEL_ID
        or request.get("model_revision") != CLIP_MODEL_REVISION
    ):
        raise ValueError("Request must use the pinned CLIP model")
    name = request["items_file"]
    if Path(name).name != name:
        raise ValueError("Request CSV must be alongside request JSON")
    csv_path = request_path.parent / name
    if digest(csv_path) != request["items_sha256"]:
        raise ValueError("Changed training feature request")
    required = keyed_csv(csv_path)
    images = keyed_csv(prepared)
    inventory = keyed_csv(embeddings / "clip_inventory.csv")
    if (
        len(required) != request["item_count"]
        or set(required) != set(images)
        or set(required) != set(inventory)
    ):
        raise ValueError("Training inventory populations differ")
    source_files = [request_path, csv_path, prepared, colors_path, outfits_path]
    source_files += [
        embeddings / name for name in ("clip_inventory.csv", "clip_embeddings.npy", "metadata.json")
    ]
    if output.exists() or any(
        output.resolve() == p.resolve() or output.resolve() in p.resolve().parents
        for p in source_files
    ):
        raise ValueError("Output exists or overlaps source components")
    before = {str(p.resolve()): digest(p) for p in source_files}
    metadata = json.loads((embeddings / "metadata.json").read_text(encoding="utf-8"))
    receipt_path = embeddings / "complete.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_files = {"clip_inventory.csv", "clip_embeddings.npy", "metadata.json"}
    if set(receipt.get("outputs", {})) != expected_files or any(
        digest(embeddings / name) != checksum for name, checksum in receipt["outputs"].items()
    ):
        raise ValueError("Embedding completion receipt checksum mismatch")
    before[str(receipt_path.resolve())] = digest(receipt_path)
    if (
        metadata["model_id"] != request["model_id"]
        or metadata["model_revision"] != request["model_revision"]
        or metadata["manifest_sha256"] != digest(prepared)
        or metadata.get("item_count") != len(required)
    ):
        raise ValueError("Embedding model or manifest provenance mismatch")
    vectors = np.load(embeddings / "clip_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (
        vectors.shape != (len(required), 512)
        or vectors.dtype != np.float32
        or not np.isfinite(vectors).all()
    ):
        raise ValueError("Invalid training embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("Non-normalized training embeddings")
    indices = [int(r["embedding_row_index"]) for r in inventory.values()]
    if sorted(indices) != list(range(len(required))):
        raise ValueError("Embedding inventory indices are not a bijection")
    palettes = {}
    with colors_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["item_id"]
            if key in palettes:
                raise ValueError(f"Duplicate color item: {key}")
            palettes[key] = row
    if set(palettes) != set(required):
        raise ValueError("Color population differs from request")
    groups = defaultdict(set)
    source_to_id = {r["source_item_id"]: k for k, r in required.items()}
    for outfit in json.loads(outfits_path.read_text(encoding="utf-8")):
        for member in outfit["items"]:
            canonical = source_to_id.get(str(member["item_id"]))
            if canonical:
                groups[canonical].add(str(outfit["set_id"]))
    if set(groups) != set(required):
        raise ValueError("Every requested item must resolve to retained source outfits")
    output.mkdir(parents=True)
    count_unmapped = 0
    with (output / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for key, source in sorted(required.items()):
            image, vector_row, color = images[key], inventory[key], palettes[key]
            if (
                source["source_split"] != "train"
                or image["target_split"] != "train"
                or vector_row["target_split"] != "train"
                or color["target_split"] != "train"
            ):
                raise ValueError(f"Cross-split training component: {key}")
            if (
                source["image_sha256"] != image["source_image_sha256"]
                or image["image_sha256"] != vector_row["image_sha256"]
                or image["image_sha256"] != color.get("image_sha256")
                or color.get("status") != "ok"
                or color.get("extractor_version") != "wardiq.color.v1"
            ):
                raise ValueError(f"Changed, missing or failed image/color component: {key}")
            common = source["common_category"] or None
            count_unmapped += common is None
            record = {
                "schema_version": "wardiq.pair-feature-item.v1",
                "item_id": key,
                "source_dataset": "Polyvore Outfits",
                "category": {
                    "source_category_id": source["source_category_id"],
                    "source_label": source["source_label"],
                    "common_category": common,
                    "mapping_status": source["mapping_status"],
                    "category_schema_version": "1.0.0",
                    "confidence": None,
                },
                "attributes": {
                    "vocabulary_source": None,
                    "vocabulary_version": None,
                    "scores": None,
                    "mask": None,
                },
                "color": {
                    "model": "LAB_KMeans",
                    "model_version": "wardiq.color.v1",
                    "palette": color["color"]["palette"],
                },
                "embedding": {
                    "model": metadata["model_id"],
                    "model_version": metadata["model_revision"],
                    "dimension": 512,
                    "provisional_dimension": True,
                    "vector": vectors[int(vector_row["embedding_row_index"])].tolist(),
                },
                "split": {"target_split": "train"},
                "provenance": {
                    "source_image_sha256": source["image_sha256"],
                    "prepared_image_sha256": image["image_sha256"],
                    "source_outfit_ids": sorted(groups[key]),
                },
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    if any(digest(p) != checksum for p, checksum in before.items()):
        raise ValueError("Training components changed during assembly")
    result = {
        "schema_version": "wardiq.pair-feature-item.v1",
        "items": len(required),
        "unmapped_common_categories": count_unmapped,
        "attributes_available": 0,
        "input_files_sha256": before,
        "records_sha256": digest(output / "records.jsonl"),
        "assembler_sha256": digest(__file__),
        "accepted_m2_cache": False,
        "status": "development_feature_view_pending_review",
    }
    write_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--colors", required=True, type=Path)
    parser.add_argument("--outfits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            assemble(
                args.request, args.prepared, args.embeddings, args.colors, args.outfits, args.output
            ),
            indent=2,
        )
    )

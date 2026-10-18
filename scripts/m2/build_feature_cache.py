"""Assemble traceable Polyvore item records from existing M2 component artifacts.

This assembler never trains or substitutes a model. Prototype inputs produce a
provisional cache; successful assembly is distinct from milestone acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from apply_taxonomy import load_taxonomy, map_category  # noqa: E402
from research_cli import protect_evidence  # noqa: E402
from validate_item_representation import validate_record  # noqa: E402

from wardiq.representation.clip_embeddings import (  # noqa: E402
    CLIP_MODEL_ID,
    CLIP_MODEL_REVISION,
    ClipEmbeddingMetadata,
    extract_clip_embeddings,
    validate_embedding_artifact,
    validate_embedding_inventory,
)

CACHE_VERSION = "wardiq.polyvore-cache.v1"
SOURCE = "Polyvore Outfits"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def keyed_csv(path, source=SOURCE):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("source_dataset") == source]
    result = {}
    for row in rows:
        key = row.get("item_id")
        if not key or key in result:
            raise ValueError(f"Missing or duplicate item ID in {path}: {key}")
        result[key] = row
    return result


def check_record(record, schema):
    errors = [e.message for e in Draft7Validator(schema).iter_errors(record)]
    if errors:
        raise ValueError("; ".join(errors))
    validate_record(record, errors)
    vector = record["embedding"]["vector"]
    if (
        vector is None
        or not np.isfinite(vector).all()
        or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)
    ):
        errors.append("Embedding must be finite, present and L2 normalized")
    if record["color"]["palette"] is None:
        errors.append("Cache records require an extracted palette")
    if errors:
        raise ValueError("; ".join(errors))


def extract_current_embeddings(args):
    """Run the owner's pinned extractor and bind its results to current image bytes."""
    directory = args.embedding_dir.resolve()
    protect_evidence(directory)
    for path in (
        args.manifest,
        args.splits,
        args.garments,
        args.taxonomy,
        args.contract,
        args.palettes,
        args.output_dir,
    ):
        if directory == path.resolve() or directory in path.resolve().parents:
            raise ValueError("Embedding output directory overlaps another input or cache output")
    garments = keyed_csv(args.garments)
    selected = []
    for row in garments.values():
        path = args.image_root / (row.get("image_path") or row.get("output_path") or "")
        if path.is_file() and row.get("usable_derivative", "true").lower() in {"true", "1"}:
            selected.append(
                {
                    **row,
                    "image_path": str(path.resolve()),
                    "target_split": row.get("target_split") or row.get("source_target_split"),
                    "usable_derivative": "True",
                    "image_sha256": digest(path),
                }
            )
    if not selected:
        raise ValueError("No accessible Polyvore derivatives for embedding extraction")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "extraction_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    extract_clip_embeddings(manifest, directory, args.image_root, args.batch_size)
    if any(digest(row["image_path"]) != row["image_sha256"] for row in selected):
        raise ValueError("Images changed during extraction; discard these embedding outputs")


def build(args):
    import pandas as pd

    output = args.output_dir.resolve()
    protect_evidence(output)
    # A mistaken output path must never overwrite inputs, including local artifacts.
    inputs = [
        args.manifest,
        args.splits,
        args.garments,
        args.taxonomy,
        args.contract,
        args.palettes,
        args.embedding_dir,
    ]
    if any(output == p.resolve() or output in p.resolve().parents for p in inputs):
        raise ValueError("Output directory overlaps an input")
    rows = keyed_csv(args.manifest)
    if not rows:
        raise ValueError("No Polyvore items in input inventory")
    splits = keyed_csv(args.splits)
    garments = keyed_csv(args.garments)
    if set(garments) - set(rows):
        raise ValueError("Garment manifest contains Polyvore IDs outside input inventory")
    missing = set(rows) - set(splits)
    if missing:
        raise ValueError(f"Missing canonical splits for {len(missing)} items")
    groups = {}
    for key in rows:
        split = splits[key]
        group = split["group_id"].split(":", 1)[0]
        if not group or split["target_split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Missing split or leakage group: {key}")
        if groups.setdefault(group, split["target_split"]) != split["target_split"]:
            raise ValueError(f"Leakage group spans splits: {group}")

    schema = read_json(args.contract)
    Draft7Validator.check_schema(schema)
    taxonomy = load_taxonomy(args.taxonomy)
    metadata_path = args.embedding_dir / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("embedding_dimension") != 512:
        raise ValueError("Pinned CLIP artifact must declare dimension 512")
    if (metadata.get("model_id"), metadata.get("model_revision")) != (
        CLIP_MODEL_ID,
        CLIP_MODEL_REVISION,
    ):
        raise ValueError("Embedding model/revision differs from the pinned M2 candidate")
    # Fixed artifact names avoid following arbitrary paths from metadata.
    vector_path = args.embedding_dir / "clip_embeddings.npy"
    inventory_path = args.embedding_dir / "clip_inventory.csv"
    vectors = np.load(vector_path, allow_pickle=False)
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    validate_embedding_artifact(vectors, ClipEmbeddingMetadata(), item_count=metadata["item_count"])
    validate_embedding_inventory(inventory, len(vectors))
    embeddings = {
        row["item_id"]: (row.to_dict(), vectors[index].tolist())
        for index, row in inventory.iterrows()
        if row["source_dataset"] in {SOURCE, "polyvore_outfits"}
    }
    palettes = {}
    for line in args.palettes.read_text(encoding="utf-8").splitlines():
        palette = json.loads(line)
        if palette["source_dataset"] == SOURCE:
            key = palette["item_id"]
            if key in palettes:
                raise ValueError(f"Duplicate palette ID: {key}")
            palettes[key] = palette
    files = [
        *inputs[:-1],
        metadata_path,
        vector_path,
        inventory_path,
        Path(__file__),
        ROOT / "scripts/validate_item_representation.py",
    ]
    provenance = {str(p.resolve()): digest(p) for p in files}
    run_signature = signature({"cache_version": CACHE_VERSION, "inputs": provenance})
    entries_dir = output / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    dispositions, records = [], []
    reused = 0
    unverified_embeddings = 0
    for key, row in sorted(rows.items()):
        garment = garments.get(key, {})
        image = args.image_root / (garment.get("image_path") or garment.get("output_path") or "")
        image_hash = digest(image) if image.is_file() else None
        fingerprint = signature([run_signature, row, garment, splits[key], image_hash])
        entry_path = entries_dir / (signature(key) + ".json")
        entry = (
            read_json(entry_path)
            if image_hash is not None and args.resume and entry_path.is_file()
            else None
        )
        if entry and entry.get("fingerprint") == fingerprint and entry.get("status") == "ok":
            check_record(entry["record"], schema)
            if entry.get("record_sha256") != signature(entry["record"]):
                raise ValueError(f"Cached record checksum mismatch: {key}")
            reused += 1
        else:
            entry = {
                "item_id": key,
                "fingerprint": fingerprint,
                "status": "unavailable",
                "reason": "no_accessible_prepared_image",
                "image_sha256": image_hash,
                "source": row,
                "split_assignment": splits[key],
                "garment": garment,
            }
            if image_hash is not None:
                try:
                    palette = palettes.get(key)
                    embedding = embeddings.get(key)
                    if not palette or palette["status"] != "ok":
                        raise ValueError("missing_or_failed_color_output")
                    if palette.get("image_sha256") != image_hash:
                        raise ValueError("stale_color_output_image_checksum")
                    if not embedding:
                        raise ValueError("missing_embedding_output")
                    embedding_row, vector = embedding
                    for component in (palette, embedding_row):
                        component_split = component.get("target_split") or component.get(
                            "source_target_split"
                        )
                        if component_split != splits[key]["target_split"]:
                            raise ValueError("component_split_mismatch")
                    # Old prototypes did not hash images. Keep their use explicit and provisional.
                    recorded_hash = embedding_row.get("image_sha256")
                    if recorded_hash and recorded_hash != image_hash:
                        raise ValueError("stale_embedding_output_image_checksum")
                    mapping = map_category(taxonomy, "polyvore_outfits", row["category_ids"])
                    if mapping is None:
                        raise ValueError("source_category_has_no_taxonomy_mapping")
                    record = {
                        "schema_version": schema["schema_version"],
                        "item_id": key,
                        "source_dataset": SOURCE,
                        "category": {
                            "source_category_id": row["category_ids"],
                            "source_label": row["category_labels"],
                            "common_category": mapping["common_category"],
                            "mapping_status": mapping["mapping_status"],
                            "category_schema_version": taxonomy["taxonomy_version"],
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
                            "model_version": palette["extractor_version"],
                            "palette": palette["color"]["palette"],
                        },
                        "embedding": {
                            "model": metadata["model_id"],
                            "model_version": metadata["model_revision"],
                            "dimension": metadata["embedding_dimension"],
                            "provisional_dimension": metadata["provisional_dimension"],
                            "vector": vector,
                        },
                        "formality": None,
                        "season": None,
                        "split": {
                            "target_split": splits[key]["target_split"],
                            "leakage_group_id": splits[key]["group_id"].split(":", 1)[0],
                        },
                    }
                    check_record(record, schema)
                    entry.update(
                        status="ok",
                        reason=None,
                        record=record,
                        record_sha256=signature(record),
                        color_provenance=palette,
                        embedding_provenance=embedding_row,
                        embedding_image_verified=bool(recorded_hash),
                    )
                except (ValueError, KeyError, TypeError) as error:
                    entry.update(status="failed", reason=str(error))
            if image_hash is not None:
                write_json(entry_path, entry)
        dispositions.append(
            {
                "item_id": key,
                "source_dataset": SOURCE,
                "source_image_id": row["source_image_id"],
                "target_split": splits[key]["target_split"],
                "group_id": splits[key]["group_id"],
                "image_path": str(image) if image_hash else "",
                "status": entry["status"],
                "reason": entry["reason"],
            }
        )
        if entry["status"] == "ok":
            records.append(entry["record"])
            unverified_embeddings += int(not entry.get("embedding_image_verified"))
        if len(dispositions) % max(1000, args.batch_size) == 0:
            print(f"Accounted for {len(dispositions)}/{len(rows)} items", flush=True)

    records_path = output / "records.jsonl"
    temporary = records_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8"
    )
    temporary.replace(records_path)
    inventory_out = output / "inventory.csv"
    with inventory_out.with_suffix(".csv.tmp").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dispositions[0]))
        writer.writeheader()
        writer.writerows(dispositions)
    inventory_out.with_suffix(".csv.tmp").replace(inventory_out)
    counts = Counter(r["status"] for r in dispositions)
    report = {
        "cache_version": CACHE_VERSION,
        "representation_version": schema["schema_version"],
        "taxonomy_version": taxonomy["taxonomy_version"],
        "attribute_vocabulary_version": None,
        "attribute_status": "unavailable_in_source_no_trained_predictor",
        "category_origin": "source_annotation_mapped_with_M1_taxonomy_confidence_null",
        "embedding_metadata": metadata,
        "embedding_images_without_checksums": unverified_embeddings,
        "configuration": {
            name: str(getattr(args, name))
            for name in (
                "manifest",
                "splits",
                "garments",
                "taxonomy",
                "contract",
                "palettes",
                "embedding_dir",
                "image_root",
                "batch_size",
            )
        },
        "run_signature": run_signature,
        "inputs": provenance,
        "inventory_items": len(rows),
        "successful": counts["ok"],
        "failed": counts["failed"],
        "unavailable": counts["unavailable"],
        "resumed": reused,
        "accessible_prepared_items": counts["ok"] + counts["failed"],
        "full_dataset_coverage_claimed": False,
        "handoff_ready": False,
        "status": "provisional",
        "m3_loading_check": "pending",
        "release_blockers": [
            "Final category predictor and component selection not approved in repository decision",
            "Taxonomy pending lead review; CLIP is a provisional integration candidate",
            "Unprepared/unavailable population and any failed items require review",
            "Cache requires downstream review before milestone acceptance",
        ],
        "outputs": {"records.jsonl": digest(records_path), "inventory.csv": digest(inventory_out)},
    }
    if unverified_embeddings:
        report["release_blockers"].append("Some embedding inputs have no per-image checksums")
    write_json(output / "manifest.json", report)
    verify_cache(output, schema)
    report["m3_loading_check"] = "passed" if records else "no_records"
    write_json(output / "manifest.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "inventory_items",
                    "successful",
                    "failed",
                    "unavailable",
                    "resumed",
                    "handoff_ready",
                )
            },
            indent=2,
        )
    )
    return int(bool(counts["failed"] or not records))


def verify_cache(output, schema):
    """Load actual serialized M3 inputs and reconcile them against the inventory."""
    report = read_json(output / "manifest.json")
    for path, expected in report["inputs"].items():
        if digest(path) != expected:
            raise ValueError(f"Cache input changed; rebuild required: {path}")
    for name, expected in report["outputs"].items():
        if name not in {"records.jsonl", "inventory.csv"} or digest(output / name) != expected:
            raise ValueError(f"Cache output checksum mismatch: {name}")
    inventory = keyed_csv(output / "inventory.csv")
    counts = Counter(row["status"] for row in inventory.values())
    if len(inventory) != report["inventory_items"] or any(
        counts[status] != report[field]
        for status, field in (
            ("ok", "successful"),
            ("failed", "failed"),
            ("unavailable", "unavailable"),
        )
    ):
        raise ValueError("Inventory counts do not reconcile")
    keys = set()
    for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        check_record(record, schema)
        key = record["item_id"]
        if key in keys or key not in inventory or inventory[key]["status"] != "ok":
            raise ValueError(f"Duplicate or unaccounted M3 record: {key}")
        entry = read_json(output / "entries" / (signature(key) + ".json"))
        if entry.get("record_sha256") != signature(record):
            raise ValueError(f"M3 record differs from checkpoint: {key}")
        if digest(inventory[key]["image_path"]) != entry["image_sha256"]:
            raise ValueError(f"Cache image changed; rebuild required: {key}")
        keys.add(key)
    if keys != {key for key, row in inventory.items() if row["status"] == "ok"}:
        raise ValueError("M3 records do not match successful inventory")
    return len(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/raw_manifest.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/split_manifest.csv")
    parser.add_argument("--garments", type=Path, default=ROOT / "artifacts/m1/garment_manifest.csv")
    parser.add_argument("--taxonomy", type=Path, default=ROOT / "configs/m1/taxonomy.json")
    parser.add_argument(
        "--contract", type=Path, default=ROOT / "configs/m2/item_representation_contract.json"
    )
    parser.add_argument(
        "--palettes", type=Path, default=ROOT / "artifacts/m2/color/v1/palettes.jsonl"
    )
    parser.add_argument(
        "--embedding-dir", type=Path, default=ROOT / "artifacts/m2/polyvore_embeddings/v1"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/m2/polyvore_cache/v1")
    parser.add_argument("--image-root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--extract-embeddings",
        action="store_true",
        help="Run the existing pinned CLIP extractor on accessible Polyvore derivatives first",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        if args.verify_only:
            count = verify_cache(args.output_dir, read_json(args.contract))
            print(f"M3 loading check: {count} records validated; inventory reconciled")
            return 0
        if args.extract_embeddings:
            extract_current_embeddings(args)
        return build(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Feature cache failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

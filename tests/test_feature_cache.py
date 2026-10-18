"""Cache integrity and restart tests using explicitly synthetic component fixtures."""

import copy
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "feature_cache", ROOT / "scripts/m2/build_feature_cache.py"
)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    source = {
        "item_id": "sample-a",
        "source_dataset": cache.SOURCE,
        "source_image_id": "123",
        "category_ids": "37",
        "category_labels": "bags",
    }
    write_csv(tmp_path / "raw.csv", [source, dict(source, item_id="sample-b")])
    split = {
        "item_id": "sample-a",
        "source_dataset": cache.SOURCE,
        "target_split": "validation",
        "group_id": "outfit:1",
    }
    write_csv(tmp_path / "splits.csv", [split, dict(split, item_id="sample-b")])
    image = tmp_path / "image.jpg"
    image.write_bytes(b"synthetic image bytes: no inference is performed by this test")
    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    write_csv(
        tmp_path / "garments.csv",
        [{"item_id": "sample-a", "source_dataset": cache.SOURCE, "image_path": str(image)}],
    )
    palette = {
        "item_id": "sample-a",
        "source_dataset": cache.SOURCE,
        "status": "ok",
        "image_sha256": image_hash,
        "target_split": "validation",
        "extractor_version": "synthetic-test-v1",
        "color": {"palette": [{"L": 50, "a": 0, "b": 0, "proportion": 1}]},
    }
    (tmp_path / "palettes.jsonl").write_text(json.dumps(palette) + "\n", encoding="utf-8")
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    vectors = np.zeros((1, 512), dtype=np.float32)
    vectors[0, 0] = 1
    np.save(embeddings / "clip_embeddings.npy", vectors)
    write_csv(
        embeddings / "clip_inventory.csv",
        [
            {
                "embedding_row_index": 0,
                "item_id": "sample-a",
                "source_dataset": "polyvore_outfits",
                "target_split": "validation",
                "image_sha256": image_hash,
            }
        ],
    )
    cache.write_json(
        embeddings / "metadata.json",
        {
            "model_id": cache.CLIP_MODEL_ID,
            "model_revision": cache.CLIP_MODEL_REVISION,
            "embedding_dimension": 512,
            "item_count": 1,
            "provisional_dimension": True,
        },
    )
    return Namespace(
        manifest=tmp_path / "raw.csv",
        splits=tmp_path / "splits.csv",
        garments=tmp_path / "garments.csv",
        palettes=tmp_path / "palettes.jsonl",
        taxonomy=ROOT / "configs/m1/taxonomy.json",
        contract=ROOT / "configs/m2/item_representation_contract.json",
        embedding_dir=embeddings,
        output_dir=tmp_path / "output",
        image_root=tmp_path,
        batch_size=1,
        resume=False,
    )


def test_assembly_resume_and_real_m3_loading(inputs):
    assert cache.build(inputs) == 0
    report = cache.read_json(inputs.output_dir / "manifest.json")
    assert (report["inventory_items"], report["successful"], report["unavailable"]) == (2, 1, 1)
    assert report["handoff_ready"] is False
    record = json.loads((inputs.output_dir / "records.jsonl").read_text())
    assert record["category"]["confidence"] is None
    assert record["attributes"]["scores"] is None
    assert record["split"]["leakage_group_id"] == "outfit"
    inputs.resume = True
    assert cache.build(inputs) == 0
    assert cache.read_json(inputs.output_dir / "manifest.json")["resumed"] == 1
    assert cache.verify_cache(inputs.output_dir, cache.read_json(inputs.contract)) == 1


def test_changed_image_does_not_resume_stale_features(inputs):
    cache.build(inputs)
    (inputs.image_root / "image.jpg").write_bytes(b"changed image")
    inputs.resume = True
    assert cache.build(inputs) == 1
    report = cache.read_json(inputs.output_dir / "manifest.json")
    assert report["successful"] == 0 and report["failed"] == 1
    assert "stale_color_output" in (inputs.output_dir / "inventory.csv").read_text()


def test_changed_model_or_invalid_vector_is_rejected(inputs):
    metadata = cache.read_json(inputs.embedding_dir / "metadata.json")
    metadata["model_revision"] = "unapproved"
    cache.write_json(inputs.embedding_dir / "metadata.json", metadata)
    with pytest.raises(ValueError, match="pinned M2 candidate"):
        cache.build(inputs)
    metadata["model_revision"] = cache.CLIP_MODEL_REVISION
    cache.write_json(inputs.embedding_dir / "metadata.json", metadata)
    np.save(inputs.embedding_dir / "clip_embeddings.npy", np.zeros((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="unit norm"):
        cache.build(inputs)


def test_duplicate_inventory_and_split_leakage_rejected(inputs):
    with inputs.manifest.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    write_csv(inputs.manifest, [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        cache.build(inputs)
    write_csv(inputs.manifest, rows)
    with inputs.splits.open(encoding="utf-8") as stream:
        splits = list(csv.DictReader(stream))
    splits[1]["target_split"] = "test"
    write_csv(inputs.splits, splits)
    with pytest.raises(ValueError, match="spans splits"):
        cache.build(inputs)


def test_corrupted_cache_is_rejected(inputs):
    cache.build(inputs)
    (inputs.output_dir / "records.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        cache.verify_cache(inputs.output_dir, cache.read_json(inputs.contract))


def test_missing_embedding_is_explicit_failure(inputs):
    write_csv(
        inputs.embedding_dir / "clip_inventory.csv",
        [
            {
                "embedding_row_index": 0,
                "item_id": "another-item",
                "source_dataset": "polyvore_outfits",
                "target_split": "validation",
            }
        ],
    )
    assert cache.build(inputs) == 1
    assert "missing_embedding_output" in (inputs.output_dir / "inventory.csv").read_text()


def test_outputs_cannot_overwrite_inputs(inputs):
    inputs.output_dir = inputs.manifest.parent
    with pytest.raises(ValueError, match="overlaps"):
        cache.build(inputs)


@pytest.mark.parametrize("problem", ["split", "palette", "embedding_hash"])
def test_bad_component_records_fail_explicitly(inputs, problem):
    if problem == "embedding_hash":
        path = inputs.embedding_dir / "clip_inventory.csv"
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["image_sha256"] = "different-image"
        write_csv(path, rows)
    else:
        palette = json.loads(inputs.palettes.read_text())
        if problem == "split":
            palette["target_split"] = "test"
        else:
            palette["color"]["palette"][0]["proportion"] = 0.2
        inputs.palettes.write_text(json.dumps(palette) + "\n", encoding="utf-8")
    assert cache.build(inputs) == 1
    assert cache.read_json(inputs.output_dir / "manifest.json")["failed"] == 1


def test_verify_detects_source_changes(inputs):
    cache.build(inputs)
    inputs.palettes.write_text("changed component output", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        cache.verify_cache(inputs.output_dir, cache.read_json(inputs.contract))


def test_m3_consumer_runs_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/m3_consumer_example.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "polyvore_outfits_disjoint_validation_114380093" in result.stdout
    assert len(result.stdout.splitlines()) == 3


@pytest.mark.parametrize(
    "problem",
    ["missing_field", "extra_field", "wrong_type", "nan", "boolean_score", "unavailable_score"],
)
def test_public_representation_validator_enforces_frozen_contract(inputs, problem):
    cache.build(inputs)
    record = copy.deepcopy(json.loads((inputs.output_dir / "records.jsonl").read_text()))
    if problem == "missing_field":
        del record["color"]["model_version"]
    elif problem == "extra_field":
        record["undocumented_field"] = True
    elif problem == "wrong_type":
        record["attributes"]["scores"] = []
    elif problem == "nan":
        record["embedding"]["vector"][0] = float("nan")
    elif problem == "boolean_score":
        record["category"]["confidence"] = True
    else:
        record["attributes"] = {
            "vocabulary_source": "fashionpedia_v1_partial",
            "vocabulary_version": "test-v1",
            "scores": {"example": 0.8},
            "mask": {"example": "unavailable_in_source"},
        }
    errors = []
    cache.validate_record(record, errors)
    assert errors, f"Invalid record accepted: {problem}"

"""Training component joins must preserve IDs, missingness and provenance."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/m3"))
import assemble_feature_items as assembly  # noqa: E402
import build_pair_features as cli  # noqa: E402


def csv_file(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def inputs(tmp_path):
    rows = [
        {
            "item_id": key,
            "source_item_id": str(i),
            "source_split": "train",
            "image_sha256": "source" + key,
            "common_category": "top" if i == 1 else "",
            "source_category_id": "19",
            "source_label": "tops",
            "mapping_status": "direct" if i == 1 else "unmapped",
        }
        for i, key in enumerate(["a", "b"], 1)
    ]
    request_csv = csv_file(tmp_path / "required.csv", rows)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "version": "wardiq.m3.training-feature-request.v1",
                "items_file": request_csv.name,
                "items_sha256": assembly.digest(request_csv),
                "item_count": 2,
                "model_id": cli.features.MODEL,
                "model_revision": cli.features.REVISION,
            }
        )
    )
    prepared = csv_file(
        tmp_path / "prepared.csv",
        [
            {
                "item_id": key,
                "target_split": "train",
                "image_sha256": "prepared" + key,
                "source_image_sha256": "source" + key,
            }
            for key in ("a", "b")
        ],
    )
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    csv_file(
        embeddings / "clip_inventory.csv",
        [
            {
                "item_id": key,
                "embedding_row_index": i,
                "target_split": "train",
                "image_sha256": "prepared" + key,
            }
            for i, key in enumerate(("a", "b"))
        ],
    )
    vectors = np.zeros((2, 512), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    np.save(embeddings / "clip_embeddings.npy", vectors)
    (embeddings / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": cli.features.MODEL,
                "model_revision": cli.features.REVISION,
                "manifest_sha256": assembly.digest(prepared),
                "item_count": 2,
            }
        )
    )
    (embeddings / "complete.json").write_text(
        json.dumps(
            {
                "outputs": {
                    name: assembly.digest(embeddings / name)
                    for name in ("clip_inventory.csv", "clip_embeddings.npy", "metadata.json")
                }
            }
        )
    )
    palettes = tmp_path / "palettes.jsonl"
    palettes.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": key,
                    "target_split": "train",
                    "image_sha256": "prepared" + key,
                    "status": "ok",
                    "extractor_version": "wardiq.color.v1",
                    "color": {"palette": [{"L": 50, "a": 0, "b": 0, "proportion": 1}]},
                }
            )
            + "\n"
            for key in ("a", "b")
        )
    )
    outfits = tmp_path / "outfits.json"
    outfits.write_text(
        json.dumps(
            [
                {"set_id": "outfit1", "items": [{"item_id": "1"}, {"item_id": "2"}]},
                {"set_id": "outfit2", "items": [{"item_id": "1"}]},
            ]
        )
    )
    return request, prepared, embeddings, palettes, outfits


def test_full_component_join_preserves_unknowns_groups_and_encoder_loading(tmp_path):
    args = inputs(tmp_path)
    result = assembly.assemble(*args, tmp_path / "out")
    records = [
        json.loads(line) for line in (tmp_path / "out/records.jsonl").read_text().splitlines()
    ]
    assert result["items"] == 2 and result["unmapped_common_categories"] == 1
    assert result["accepted_m2_cache"] is False
    assert records[0]["provenance"]["source_outfit_ids"] == ["outfit1", "outfit2"]
    assert records[1]["category"]["common_category"] is None
    assert records[0]["embedding"]["vector"][0] == 1
    assert records[1]["embedding"]["vector"][1] == 1
    obj = cli.features.PairFeatureBuilder(cli.features.make_config(["jeans"]), "train")
    for record in records:
        assert record["attributes"]["scores"] is None
        obj.add_item(record)
    assert np.isfinite(obj.transform_pair("a", "b")).all()


@pytest.mark.parametrize(
    "case", ["request", "color", "rows", "groups", "vector", "model", "output"]
)
def test_reject_corrupt_components(tmp_path, case):
    args = inputs(tmp_path)
    if case == "request":
        (tmp_path / "required.csv").write_text("changed")
    elif case == "color":
        palettes = args[3]
        palettes.write_text(palettes.read_text().replace('"status": "ok"', '"status": "failed"'))
    elif case == "rows":
        inventory = args[2] / "clip_inventory.csv"
        inventory.write_text(inventory.read_text().replace("b,1", "b,0"))
    elif case == "groups":
        args[4].write_text("[]")
    elif case == "vector":
        np.save(args[2] / "clip_embeddings.npy", np.zeros((2, 512), dtype=np.float32))
    elif case == "model":
        path = args[2] / "metadata.json"
        data = json.loads(path.read_text())
        data["model_revision"] = "other"
        path.write_text(json.dumps(data))
    elif case == "output":
        (tmp_path / "out").mkdir()
    if case in {"rows", "vector", "model"}:
        # An internally checksummed upstream artifact can still be semantically invalid.
        # Exercise the semantic validation, separately from stale-file detection.
        (args[2] / "complete.json").write_text(
            json.dumps(
                {
                    "outputs": {
                        name: assembly.digest(args[2] / name)
                        for name in ("clip_inventory.csv", "clip_embeddings.npy", "metadata.json")
                    }
                }
            )
        )
    with pytest.raises(ValueError):
        assembly.assemble(*args, tmp_path / "out")

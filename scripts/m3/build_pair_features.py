"""Export aligned pair matrices using the same transformation as online scoring."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from extract_training_features import digest, write_json

MODULE = Path(__file__).resolve().parents[2] / "src/wardiq/compatibility/pair_features.py"
SPEC = importlib.util.spec_from_file_location("pair_features", MODULE)
features = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(features)


def build(records, pairs_path, config_path, output, split):
    records, pairs_path, config_path, output = map(Path, (records, pairs_path, config_path, output))
    protected = [records.resolve(), pairs_path.resolve(), config_path.resolve()]
    if output.exists() or any(
        output.resolve() == p or output.resolve() in p.parents for p in protected
    ):
        raise ValueError("Output exists or overlaps inputs; select a new version directory")
    before = {str(p): digest(p) for p in protected}
    config = json.loads(config_path.read_text())
    builder = features.PairFeatureBuilder(config, split)
    needed, seen, count = set(), set(), 0
    with pairs_path.open(encoding="utf-8") as stream:
        for line in stream:
            pair = json.loads(line)
            if pair.get("split") != split:
                raise ValueError("Pair split mismatch")
            if type(pair.get("label")) is not int or pair["label"] not in (0, 1):
                raise ValueError("Expected observed binary pair labels")
            key = tuple(sorted([pair["left_item_id"], pair["right_item_id"]]))
            if key[0] == key[1] or key in seen:
                raise ValueError(f"Self or duplicate unordered pair: {key}")
            seen.add(key)
            needed.update(key)
            count += 1
    if count == 0:
        raise ValueError("Empty pair population")
    with records.open(encoding="utf-8") as stream:
        record_ids = set()
        for line in stream:
            record = json.loads(line)
            key = record["item_id"]
            if key in record_ids:
                raise ValueError(f"Duplicate cache record: {key}")
            record_ids.add(key)
            if key in needed:
                builder.add_item(record)
    missing = needed - set(builder.items)
    if missing:
        raise ValueError(f"Unresolved item references: {len(missing)}; first={sorted(missing)[:5]}")
    output.mkdir(parents=True)
    matrix = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype="float32", shape=(count, config["dimension"])
    )
    labels = np.lib.format.open_memmap(
        output / "labels.npy", mode="w+", dtype="int8", shape=(count,)
    )
    with (
        pairs_path.open(encoding="utf-8") as stream,
        (output / "pair_inventory.csv").open("w", encoding="utf-8", newline="") as inventory,
    ):
        writer = csv.writer(inventory)
        writer.writerow(
            ["row_index", "left_item_id", "right_item_id", "split", "label", "label_origin"]
        )
        for index, line in enumerate(stream):
            pair = json.loads(line)
            matrix[index] = builder.transform_pair(pair["left_item_id"], pair["right_item_id"])
            labels[index] = pair["label"]
            writer.writerow(
                [
                    index,
                    pair["left_item_id"],
                    pair["right_item_id"],
                    split,
                    pair["label"],
                    pair.get("label_origin", "unknown"),
                ]
            )
            if (index + 1) % 10000 == 0:
                print(f"Features {split}: {index + 1}/{count}", flush=True)
    matrix.flush()
    labels.flush()
    del matrix, labels
    if any(digest(path) != checksum for path, checksum in before.items()):
        raise ValueError("Input changed during feature generation")
    write_json(output / "feature_schema.json", config)
    result = {
        "version": features.VERSION,
        "status": "development_pending_protocol_review",
        "split": split,
        "rows": count,
        "required_items": len(needed),
        "resolved_items": len(builder.items),
        "unresolved_items": 0,
        "dimension": config["dimension"],
        "config_sha256": features.fingerprint(config),
        "input_files_sha256": before,
        "builder_sha256": digest(MODULE),
        "fitted_preprocessing": False,
        "training_ready_approved": False,
        "outputs_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    write_json(output / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    args = parser.parse_args()
    build(args.records, args.pairs, args.config, args.output, args.split)

"""Lightweight validator for configs/m1/item_output_schema.json records.

No external dependencies (does not require the `jsonschema` package) so every
component owner can run it with a stock Python 3 install.

Usage (from the repository root):
    python scripts/validate_item_output.py data/samples/item_output/examples.json
    python scripts/validate_item_output.py --record '{"item_id": "...", ...}'

Exit code is 1 if any record fails. Prints one line per record: OK or a list
of specific reasons.
"""
import argparse
import json
import sys

COMMON_CATEGORIES = {"top", "bottom", "one_piece", "outerwear", "footwear", "bag",
                      "headwear", "eyewear", "scarf", "jewellery", "accessory_other"}
SOURCES = {"Fashion-MNIST", "Fashionpedia", "Polyvore Outfits", "DeepFashion-MultiModal"}
MAPPING_STATUSES = {"direct", "approximate", "ambiguous", "unmapped", "not_a_garment"}
ATTR_STATUSES = {"present", "not_annotated", "unavailable_in_source"}
SPLITS = {"train", "validation", "test", None}


def err(msg, errors):
    errors.append(msg)


def validate_record(r, errors):
    if not isinstance(r, dict):
        return err("record is not a JSON object", errors)

    if r.get("schema_version") != "1.0.0":
        err(f"schema_version must be '0.1.0', got {r.get('schema_version')!r}", errors)

    if not isinstance(r.get("item_id"), str) or not r["item_id"]:
        err("item_id must be a non-empty string", errors)

    if r.get("source_dataset") not in SOURCES:
        err(f"source_dataset {r.get('source_dataset')!r} not in {sorted(SOURCES)}", errors)

    cat = r.get("category", {})
    if not isinstance(cat, dict):
        err("category must be an object", errors)
    else:
        if cat.get("mapping_status") not in MAPPING_STATUSES:
            err(f"category.mapping_status invalid: {cat.get('mapping_status')!r}", errors)
        cc = cat.get("common_category")
        if cc is not None and cc not in COMMON_CATEGORIES:
            err(f"category.common_category {cc!r} not in the 11-category vocabulary", errors)
        if cat.get("mapping_status") in ("unmapped", "not_a_garment") and cc is not None:
            err("category.common_category must be null when mapping_status is unmapped/not_a_garment", errors)
        conf = cat.get("confidence")
        if conf is not None and not (isinstance(conf, (int, float)) and 0 <= conf <= 1):
            err(f"category.confidence must be null or in [0,1], got {conf!r}", errors)
        if not isinstance(cat.get("category_schema_version"), str) or not cat["category_schema_version"]:
            err("category.category_schema_version must be a non-empty string", errors)

    attrs = r.get("attributes", {})
    if not isinstance(attrs, dict):
        err("attributes must be an object", errors)
    else:
        scores, mask = attrs.get("scores"), attrs.get("mask")
        if (scores is None) != (mask is None):
            err("attributes.scores and attributes.mask must both be null or both be set", errors)
        if scores is not None:
            names = list(scores.keys())
            if names != sorted(names):
                err("attributes.scores keys must be in strict ascending alphabetical order", errors)
            if set(scores) != set(mask or {}):
                err("attributes.scores and attributes.mask must have identical keys", errors)
            for k, v in scores.items():
                if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 1):
                    err(f"attributes.scores[{k!r}] must be null or in [0,1], got {v!r}", errors)
                st = (mask or {}).get(k)
                if st not in ATTR_STATUSES:
                    err(f"attributes.mask[{k!r}] invalid status {st!r}", errors)
                if st == "not_annotated" and v is not None:
                    err(f"attributes.scores[{k!r}] must be null when mask says not_annotated (never guess a value)", errors)

    color = r.get("color", {})
    if not isinstance(color, dict):
        err("color must be an object", errors)
    else:
        palette = color.get("palette")
        if palette is not None:
            if not (1 <= len(palette) <= 8):
                err("color.palette must have 1-8 entries", errors)
            total = 0.0
            for i, sw in enumerate(palette):
                for k, lo, hi in (("L", 0, 100), ("a", -128, 127), ("b", -128, 127), ("proportion", 0, 1)):
                    v = sw.get(k)
                    if not (isinstance(v, (int, float)) and lo <= v <= hi):
                        err(f"color.palette[{i}].{k} must be in [{lo},{hi}], got {v!r}", errors)
                total += sw.get("proportion", 0) or 0
            if abs(total - 1.0) > 0.01:
                err(f"color.palette proportions must sum to ~1.0, got {total:.4f}", errors)

    emb = r.get("embedding", {})
    if not isinstance(emb, dict):
        err("embedding must be an object", errors)
    else:
        dim, vec = emb.get("dimension"), emb.get("vector")
        if "provisional_dimension" not in emb or not isinstance(emb["provisional_dimension"], bool):
            err("embedding.provisional_dimension must be a boolean and must be present", errors)
        if vec is not None:
            if dim is None:
                err("embedding.vector is set but embedding.dimension is null", errors)
            elif len(vec) != dim:
                err(f"embedding.vector length {len(vec)} != embedding.dimension {dim}", errors)

    for field in ("formality", "season"):
        if r.get(field) is not None:
            err(f"{field} must be null in M1 (not populated / not validated)", errors)

    split = r.get("split", {})
    if not isinstance(split, dict) or split.get("target_split") not in SPLITS:
        err(f"split.target_split invalid: {split.get('target_split') if isinstance(split, dict) else split!r}", errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="JSON file: a single record object or a list of records.")
    ap.add_argument("--record", help="A single record as a JSON string, instead of --path.")
    a = ap.parse_args()

    if a.record:
        records = [json.loads(a.record)]
    elif a.path:
        data = json.loads(open(a.path, encoding="utf-8").read())
        if isinstance(data, dict) and ("valid_examples" in data or "invalid_examples" in data):
            any_fail = False
            for r in data.get("valid_examples", []):
                errors = []
                validate_record(r, errors)
                label = r.get("item_id", "?")
                if errors:
                    any_fail = True
                    print(f"FAIL (expected OK)   {label}")
                    for e in errors:
                        print(f"        - {e}")
                else:
                    print(f"OK                    {label}")
            for ex in data.get("invalid_examples", []):
                r = ex["record"]
                errors = []
                validate_record(r, errors)
                label = r.get("item_id", "?")
                if errors:
                    print(f"CORRECTLY REJECTED    {label}: {ex.get('reason', '')}")
                else:
                    any_fail = True
                    print(f"FAIL (expected REJECT) {label}: validator missed a deliberately invalid record")
            sys.exit(1 if any_fail else 0)
        records = data if isinstance(data, list) else [data]
    else:
        ap.error("provide a path or --record")

    any_fail = False
    for i, r in enumerate(records):
        errors = []
        validate_record(r, errors)
        label = r.get("item_id", f"record[{i}]") if isinstance(r, dict) else f"record[{i}]"
        if errors:
            any_fail = True
            print(f"FAIL  {label}")
            for e in errors:
                print(f"        - {e}")
        else:
            print(f"OK    {label}")

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()

"""Symmetric, masked pair features for the provisional M3 baseline.

No fitting, labels or split statistics enter the feature transformation.
Distances are descriptors, not calibrated compatibility probabilities.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
VERSION = "wardiq.pair-features.v1"
CATEGORIES = sorted(
    [
        "top",
        "bottom",
        "one_piece",
        "outerwear",
        "footwear",
        "bag",
        "headwear",
        "eyewear",
        "scarf",
        "jewellery",
        "accessory_other",
    ]
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def make_config(attribute_names: list[str]) -> dict[str, Any]:
    """Freeze vocabulary explicitly; never discover it from validation/test data."""
    if len(set(attribute_names)) != len(attribute_names):
        raise ValueError("Duplicate attribute names")
    if any(not isinstance(x, str) or not x for x in attribute_names):
        raise ValueError("Attribute names must be nonempty strings")
    names = sorted(attribute_names)
    categories = CATEGORIES + ["unknown"]
    pairs = list(itertools.combinations_with_replacement(categories, 2))
    columns = [f"category:{a}|{b}" for a, b in pairs]
    columns += ["category:both_mapped", "clip:cosine"]
    columns += [f"clip:abs_difference:{i}" for i in range(512)]
    columns += [f"clip:product:{i}" for i in range(512)]
    columns += [
        "lab:mean_delta_e76_over_100",
        "lab:expected_delta_e76_over_100",
        "lab:lightness_difference_over_100",
        "lab:chroma_difference_over_181",
        "lab:both_available",
    ]
    for name in names:
        columns += [
            f"attribute:{name}:abs_difference",
            f"attribute:{name}:product",
            f"attribute:{name}:both_observed",
        ]
    return {
        "version": VERSION,
        "accepted_item_schema_versions": ["1.0.0", "wardiq.pair-feature-item.v1"],
        "taxonomy_version": "1.0.0",
        "embedding_model": MODEL,
        "embedding_revision": REVISION,
        "embedding_dimension": 512,
        "attribute_vocabulary_source": "fashionpedia_v1_partial",
        "attribute_vocabulary_version": "1.0.0",
        "attributes": names,
        "category_pairs": [list(p) for p in pairs],
        "feature_names": columns,
        "dimension": len(columns),
        "dtype": "float32",
        "symmetric": True,
        "fitted_preprocessing": False,
        "status": "development_pending_review",
    }


class PairFeatureBuilder:
    """Prepare each item once, then use identical features in training/scoring."""

    def __init__(self, config: dict[str, Any], split: str) -> None:
        if config != make_config(config.get("attributes", [])):
            raise ValueError("Unsupported or changed feature configuration")
        if split not in {"train", "validation", "test", "inference"}:
            raise ValueError("Expected explicit train/validation/test split")
        self.config = config
        self.split = split
        self.items: dict[str, dict[str, Any]] = {}
        self.category_positions = {
            tuple(sorted(p)): i for i, p in enumerate(config["category_pairs"])
        }

    def add_item(self, record: dict[str, Any]) -> None:
        key = record.get("item_id")
        if not isinstance(key, str) or not key or key in self.items:
            raise ValueError(f"Missing or duplicate item ID: {key}")
        if record.get("schema_version") not in self.config["accepted_item_schema_versions"]:
            raise ValueError(f"Item schema mismatch: {key}")
        expected_split = None if self.split == "inference" else self.split
        if record.get("split", {}).get("target_split") != expected_split:
            raise ValueError(f"Cross-split item: {key}")
        category = record["category"]
        if category.get("category_schema_version") != self.config["taxonomy_version"]:
            raise ValueError(f"Taxonomy version mismatch: {key}")
        common = category.get("common_category")
        if common is not None and common not in CATEGORIES:
            raise ValueError(f"Unsupported common category: {key}: {common}")
        if category.get("mapping_status") in {"unmapped", "not_a_garment"} and common:
            raise ValueError(f"Inconsistent category missingness: {key}")
        embedding = record["embedding"]
        if (
            embedding.get("model") != MODEL
            or embedding.get("model_version") != REVISION
            or embedding.get("dimension") != 512
        ):
            raise ValueError(f"Embedding model/revision/dimension mismatch: {key}")
        vector = np.asarray(embedding["vector"], dtype=np.float32)
        if (
            vector.shape != (512,)
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4)
        ):
            raise ValueError(f"Invalid normalized embedding: {key}")
        color = record["color"]
        palette = color.get("palette")
        lab = weights = None
        if palette is not None:
            if (
                color.get("model") != "LAB_KMeans"
                or color.get("model_version") != "wardiq.color.v1"
            ):
                raise ValueError(f"Color model/version mismatch: {key}")
            lab = np.asarray([[x["L"], x["a"], x["b"]] for x in palette], dtype=np.float64)
            weights = np.asarray([x["proportion"] for x in palette], dtype=np.float64)
            if (
                lab.ndim != 2
                or lab.shape[1:] != (3,)
                or not 1 <= len(lab) <= 8
                or not np.isfinite(lab).all()
                or not np.isfinite(weights).all()
                or (weights < 0).any()
                or (weights > 1).any()
                or not np.isclose(weights.sum(), 1, atol=0.01)
                or (lab[:, 0] < 0).any()
                or (lab[:, 0] > 100).any()
                or (lab[:, 1:] < -128).any()
                or (lab[:, 1:] > 127).any()
            ):
                raise ValueError(f"Invalid LAB palette: {key}")
            weights = weights / weights.sum()
        attrs = record["attributes"]
        scores, masks = attrs.get("scores"), attrs.get("mask")
        if (scores is None) != (masks is None):
            raise ValueError(f"Attribute score/mask mismatch: {key}")
        if scores is not None:
            if (
                set(scores) != set(masks)
                or list(scores) != sorted(scores)
                or set(scores) - set(self.config["attributes"])
                or attrs.get("vocabulary_source") != self.config["attribute_vocabulary_source"]
                or str(attrs.get("vocabulary_version"))
                != self.config["attribute_vocabulary_version"]
            ):
                raise ValueError(f"Attribute vocabulary/order mismatch: {key}")
            for name, score in scores.items():
                mask = masks[name]
                if (
                    mask not in {"present", "not_annotated", "unavailable_in_source"}
                    or (mask != "present" and score is not None)
                    or (mask == "present" and score is None)
                    or (
                        score is not None
                        and (
                            isinstance(score, bool) or not np.isfinite(score) or not 0 <= score <= 1
                        )
                    )
                ):
                    raise ValueError(f"Invalid attribute score/mask: {key}: {name}")
        values = np.zeros(len(self.config["attributes"]), dtype=np.float32)
        observed = np.zeros_like(values)
        for i, name in enumerate(self.config["attributes"]):
            if scores is not None and masks.get(name) == "present":
                values[i] = scores[name]
                observed[i] = 1
        self.items[key] = {
            "category": common or "unknown",
            "vector": vector,
            "lab": lab,
            "weights": weights,
            "attributes": values,
            "observed": observed,
        }

    def transform_pair(self, left_id: str, right_id: str) -> NDArray[np.float32]:
        if left_id == right_id:
            raise ValueError(f"Self pair: {left_id}")
        missing = [x for x in (left_id, right_id) if x not in self.items]
        if missing:
            raise ValueError(f"Unresolved pair item IDs: {missing}")
        # Canonical order also ensures byte-exact floating-point symmetry.
        left, right = [self.items[k] for k in sorted([left_id, right_id])]
        category = np.zeros(len(self.category_positions), dtype=np.float32)
        category[self.category_positions[tuple(sorted([left["category"], right["category"]]))]] = 1
        a, b = left["vector"], right["vector"]
        colors = np.zeros(5, dtype=np.float32)
        if left["lab"] is not None and right["lab"] is not None:
            left_lab, right_lab = left["lab"], right["lab"]
            lw, rw = left["weights"], right["weights"]
            lm, rm = lw @ left_lab, rw @ right_lab
            distances = np.linalg.norm(left_lab[:, None, :] - right_lab[None, :, :], axis=2)
            colors[:] = [
                np.linalg.norm(lm - rm) / 100,
                lw @ distances @ rw / 100,
                abs(lm[0] - rm[0]) / 100,
                abs(np.linalg.norm(lm[1:]) - np.linalg.norm(rm[1:])) / 181,
                1,
            ]
        joint = left["observed"] * right["observed"]
        attributes = np.column_stack(
            (
                np.abs(left["attributes"] - right["attributes"]) * joint,
                left["attributes"] * right["attributes"] * joint,
                joint,
            )
        ).ravel()
        mapped = float(left["category"] != "unknown" and right["category"] != "unknown")
        result = np.concatenate(
            (category, [mapped, np.dot(a, b)], np.abs(a - b), a * b, colors, attributes)
        ).astype(np.float32)
        if result.shape != (self.config["dimension"],) or not np.isfinite(result).all():
            raise ValueError("Invalid feature output")
        return result

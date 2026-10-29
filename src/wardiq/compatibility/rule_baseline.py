"""Provisional rule-based pair compatibility baseline.

Weights and scoring semantics are experimental and require team approval.
"""

from math import exp, isfinite
from typing import Final, Sequence

DEFAULT_WEIGHTS: Final[dict[str, float]] = {
    "clip:cosine": 0.7,
    "lab:expected_delta_e76_over_100": 0.3,
}


def score_pair(
    *,
    clip_cosine: float,
    color_distance: float | None,
) -> float:
    """Return a heuristic score in [0, 1], not a probability.

    CLIP cosine is mapped from [-1, 1] to [0, 1].
    Normalized LAB distance is converted using exp(-distance).

    Missing color uses the CLIP score alone. This changes the
    effective weighting and is provisional, pending team review.
    """

    if not isfinite(clip_cosine) or not -1.0 <= clip_cosine <= 1.0:
        raise ValueError("clip_cosine must be finite and in [-1, 1]")

    if color_distance is not None:
        if not isfinite(color_distance) or color_distance < 0.0:
            raise ValueError("color_distance must be finite and non-negative")

    visual_score = (clip_cosine + 1.0) / 2.0

    if color_distance is None:
        return visual_score

    color_score = exp(-color_distance)

    return (
        DEFAULT_WEIGHTS["clip:cosine"] * visual_score
        + DEFAULT_WEIGHTS["lab:expected_delta_e76_over_100"] * color_score
    )


def score_pair_features(
    *,
    feature_names: Sequence[str],
    feature_values: Sequence[float],
) -> float:
    """Score the named features returned by PairFeatureBuilder."""

    if len(feature_names) != len(feature_values):
        raise ValueError("Feature names and values must have equal lengths")

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique")

    if not all(isfinite(value) for value in feature_values):
        raise ValueError("Feature values must be finite")

    features = dict(zip(feature_names, feature_values))

    required = {
        "clip:cosine",
        "lab:expected_delta_e76_over_100",
        "lab:both_available",
    }
    if not required.issubset(features):
        raise ValueError("Required compatibility features are missing")

    color_available = features["lab:both_available"]
    if color_available not in (0.0, 1.0):
        raise ValueError("lab:both_available must be 0 or 1")

    color_distance = features["lab:expected_delta_e76_over_100"]
    if color_available == 0.0:
        color_distance = None

    return score_pair(
        clip_cosine=features["clip:cosine"],
        color_distance=color_distance,
    )

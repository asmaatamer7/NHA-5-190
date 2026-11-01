
import pytest

from wardiq.compatibility.rule_baseline import (
    DEFAULT_WEIGHTS,
    score_pair,
    score_pair_features,
)

FEATURE_NAMES = [
    "clip:cosine",
    "lab:expected_delta_e76_over_100",
    "lab:both_available",
]


def test_default_weights_are_explicit_and_sum_to_one():
    assert DEFAULT_WEIGHTS["clip:cosine"] == 0.7
    assert DEFAULT_WEIGHTS["lab:expected_delta_e76_over_100"] == 0.3
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_missing_color_uses_clip_only_without_fake_color_penalty():
    score = score_pair(clip_cosine=0.8, color_distance=None)
    assert score == pytest.approx(0.9)


def test_available_color_contributes_to_score():
    score = score_pair(clip_cosine=0.8, color_distance=0.0)
    assert score == pytest.approx(0.93)


def test_score_pair_features_uses_shared_feature_values():
    score = score_pair_features(
        feature_names=FEATURE_NAMES,
        feature_values=[0.8, 0.0, 1.0],
    )
    assert score == pytest.approx(0.93)


def test_score_pair_features_handles_missing_color():
    score = score_pair_features(
        feature_names=FEATURE_NAMES,
        feature_values=[0.8, 0.0, 0.0],
    )
    assert score == pytest.approx(0.9)


@pytest.mark.parametrize(
    "clip_cosine",
    [float("nan"), float("inf"), -1.1, 1.1],
)
def test_invalid_clip_cosine_is_rejected(clip_cosine):
    with pytest.raises(ValueError):
        score_pair(clip_cosine=clip_cosine, color_distance=None)


@pytest.mark.parametrize("distance", [float("nan"), float("inf"), -0.1])
def test_invalid_color_distance_is_rejected(distance):
    with pytest.raises(ValueError):
        score_pair(clip_cosine=0.5, color_distance=distance)


def test_duplicate_feature_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        score_pair_features(
            feature_names=["clip:cosine", "clip:cosine", "lab:both_available"],
            feature_values=[0.5, 0.5, 0.0],
        )


def test_non_finite_feature_values_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        score_pair_features(
            feature_names=FEATURE_NAMES,
            feature_values=[float("nan"), 0.0, 1.0],
        )


def test_missing_required_features_are_rejected():
    with pytest.raises(ValueError, match="Required"):
        score_pair_features(
            feature_names=["clip:cosine"],
            feature_values=[0.5],
        )

@pytest.mark.parametrize(
    "clip_cosine",
    [float("nan"), float("inf"), -1.1, 1.1],
)
def test_feature_scoring_rejects_invalid_clip_cosine(clip_cosine):
    with pytest.raises(ValueError):
        score_pair_features(
            feature_names=FEATURE_NAMES,
            feature_values=[clip_cosine, 0.0, 1.0],
        )


def test_feature_scoring_rejects_invalid_color_availability():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        score_pair_features(
            feature_names=FEATURE_NAMES,
            feature_values=[0.5, 0.0, 0.5],
        )


def test_feature_scoring_rejects_negative_color_distance_when_available():
    with pytest.raises(ValueError, match="non-negative"):
        score_pair_features(
            feature_names=FEATURE_NAMES,
            feature_values=[0.5, -0.1, 1.0],
        )


def test_feature_scoring_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        score_pair_features(
            feature_names=["clip:cosine", "lab:both_available"],
            feature_values=[0.5],
        )


def test_feature_scoring_is_independent_of_feature_order():
    original = score_pair_features(
        feature_names=FEATURE_NAMES,
        feature_values=[0.8, 0.0, 1.0],
    )

    reordered = score_pair_features(
        feature_names=[
            "lab:both_available",
            "clip:cosine",
            "lab:expected_delta_e76_over_100",
        ],
        feature_values=[1.0, 0.8, 0.0],
    )

    assert reordered == pytest.approx(original)


def test_feature_scoring_rejects_non_finite_color_distance():
    with pytest.raises(ValueError, match="finite"):
        score_pair_features(
            feature_names=FEATURE_NAMES,
            feature_values=[0.8, float("nan"), 1.0],
        )

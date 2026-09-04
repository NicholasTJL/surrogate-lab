import numpy as np
import pandas as pd
import pytest

from surrogate_lab.core.ood import assess_ood, fit_ood_detector
from surrogate_lab.core.schema import DatasetSchema


def _schema(categorical: bool = False) -> DatasetSchema:
    if categorical:
        return DatasetSchema(features=["x1", "x2", "cat"], target="y", categorical_features=["cat"])
    return DatasetSchema(features=["x1", "x2"], target="y")


def test_point_inside_range_and_close_to_training_data_is_within_domain() -> None:
    x_train = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0, 4.0], "x2": [0.0, 1.0, 2.0, 3.0, 4.0]})
    detector = fit_ood_detector(x_train, _schema())

    assessment = assess_ood(detector, {"x1": 2.0, "x2": 2.0})

    assert assessment.within_training_domain is True
    assert assessment.warnings == []
    assert assessment.out_of_range_features == []


def test_feature_outside_training_range_is_flagged() -> None:
    x_train = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0, 4.0], "x2": [0.0, 1.0, 2.0, 3.0, 4.0]})
    detector = fit_ood_detector(x_train, _schema())

    assessment = assess_ood(detector, {"x1": 100.0, "x2": 2.0})

    assert assessment.within_training_domain is False
    assert "x1" in assessment.out_of_range_features
    assert any("x1" in w and "outside the training range" in w for w in assessment.warnings)


def test_point_far_from_all_training_points_flagged_by_nn_distance_even_in_range() -> None:
    # Two dense clusters with a gap between them: a query point placed in the gap is
    # inside the overall [min, max] range on each feature individually, but far from
    # every actual training point.
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(30, 2))
    cluster_b = rng.normal(loc=10.0, scale=0.1, size=(30, 2))
    x_train = pd.DataFrame(
        np.vstack([cluster_a, cluster_b]), columns=["x1", "x2"]
    )
    detector = fit_ood_detector(x_train, _schema(), nn_distance_percentile=0.95)

    midpoint_assessment = assess_ood(detector, {"x1": 5.0, "x2": 5.0})
    in_cluster_assessment = assess_ood(detector, {"x1": 0.0, "x2": 0.0})

    assert midpoint_assessment.out_of_range_features == []  # in range on each feature alone
    assert midpoint_assessment.within_training_domain is False
    assert any("nearest-neighbor distance" in w for w in midpoint_assessment.warnings)
    assert in_cluster_assessment.within_training_domain is True
    assert (
        midpoint_assessment.nearest_neighbor_distance
        > in_cluster_assessment.nearest_neighbor_distance
    )


def test_unseen_category_is_flagged() -> None:
    x_train = pd.DataFrame(
        {"x1": [0.0, 1.0, 2.0], "x2": [0.0, 1.0, 2.0], "cat": ["a", "b", "a"]}
    )
    detector = fit_ood_detector(x_train, _schema(categorical=True))

    assessment = assess_ood(detector, {"x1": 1.0, "x2": 1.0, "cat": "z"})

    assert assessment.within_training_domain is False
    assert assessment.unseen_categories == {"cat": "z"}


def test_single_training_point_collapses_range_but_does_not_crash() -> None:
    x_train = pd.DataFrame({"x1": [5.0], "x2": [5.0]})
    detector = fit_ood_detector(x_train, _schema())

    assert detector.nn_distance_threshold is None
    assert detector.insufficient_data_warning is not None

    same_point = assess_ood(detector, {"x1": 5.0, "x2": 5.0})
    different_point = assess_ood(detector, {"x1": 6.0, "x2": 6.0})

    # With one training point, its range is a single value: an identical point is
    # in-range. within_training_domain is still True here (no range/distance/ensemble
    # flag fired), but the informational insufficient-data warning is present in both
    # assessments, noting that no distance threshold could be calibrated.
    assert same_point.out_of_range_features == []
    assert same_point.within_training_domain is True
    assert any("insufficient" in w.lower() or "1 training row" in w for w in same_point.warnings)
    assert "x1" in different_point.out_of_range_features
    # nearest_neighbor_distance is still computed even with no threshold to compare it to.
    assert not np.isnan(different_point.nearest_neighbor_distance)
    assert different_point.nearest_neighbor_distance > 0


def test_fit_ood_detector_rejects_empty_training_set() -> None:
    x_train = pd.DataFrame({"x1": [], "x2": []})

    with pytest.raises(ValueError, match="empty training set"):
        fit_ood_detector(x_train, _schema())


def test_fit_ood_detector_rejects_invalid_percentile() -> None:
    x_train = pd.DataFrame({"x1": [1.0, 2.0], "x2": [1.0, 2.0]})

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        fit_ood_detector(x_train, _schema(), nn_distance_percentile=1.5)


def test_mahalanobis_distance_accounts_for_correlation() -> None:
    # x1, x2 strongly (but not perfectly) correlated: training data varies much more
    # along the x1=x2 diagonal than perpendicular to it. Two query points equidistant
    # from the centroid in *Euclidean* terms should not be equidistant in Mahalanobis
    # terms: the one that moves against the correlation (perpendicular) is far more
    # unusual relative to the training data's actual spread than the one that moves
    # along it.
    rng = np.random.default_rng(0)
    correlated = rng.multivariate_normal(
        mean=[0.0, 0.0], cov=[[4.0, 3.9], [3.9, 4.0]], size=200
    )
    x_train = pd.DataFrame(correlated, columns=["x1", "x2"])
    detector = fit_ood_detector(x_train, _schema(), distance_metric="mahalanobis")

    along_correlation = assess_ood(detector, {"x1": 3.0, "x2": 3.0})
    against_correlation = assess_ood(detector, {"x1": 3.0, "x2": -3.0})

    assert (
        against_correlation.nearest_neighbor_distance
        > along_correlation.nearest_neighbor_distance
    )
    assert against_correlation.distance_metric == "mahalanobis"


def test_mahalanobis_falls_back_gracefully_with_singular_covariance() -> None:
    # Only 1 training point isn't enough to estimate any covariance matrix.
    x_train = pd.DataFrame({"x1": [1.0], "x2": [1.0]})
    detector = fit_ood_detector(x_train, _schema(), distance_metric="mahalanobis")

    assert detector.covariance_inv is None
    assert detector.mahalanobis_threshold is None

    assessment = assess_ood(detector, {"x1": 2.0, "x2": 2.0})

    # Falls back to a finite Euclidean distance instead of crashing or returning NaN, even
    # though the covariance matrix couldn't be estimated. distance_metric must say "euclidean"
    # here (not "mahalanobis"), since that's what was actually computed — the label must match
    # what was actually used, not what was configured, or the report/CSV would claim a
    # Mahalanobis figure that's actually Euclidean.
    assert not np.isnan(assessment.nearest_neighbor_distance)
    assert assessment.nearest_neighbor_distance > 0
    assert assessment.nearest_neighbor_distance_threshold is None
    assert assessment.distance_metric == "euclidean"


def test_mahalanobis_with_few_training_points_uses_population_covariance() -> None:
    # n_train (2) <= number of numeric features (2): not enough degrees of freedom for
    # the usual (ddof=1) sample covariance, so a population (ddof=0) estimate is used
    # instead of crashing.
    x_train = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
    detector = fit_ood_detector(x_train, _schema(), distance_metric="mahalanobis")

    assert detector.covariance_inv is not None
    assessment = assess_ood(detector, {"x1": 1.5, "x2": 3.5})
    assert not np.isnan(assessment.nearest_neighbor_distance)


def test_ensemble_disagreement_signal_flags_high_disagreement() -> None:
    x_train = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0, 4.0], "x2": [0.0, 1.0, 2.0, 3.0, 4.0]})
    detector = fit_ood_detector(x_train, _schema())

    low_disagreement = assess_ood(
        detector,
        {"x1": 2.0, "x2": 2.0},
        ensemble_disagreement=0.1,
        ensemble_disagreement_threshold=1.0,
    )
    high_disagreement = assess_ood(
        detector,
        {"x1": 2.0, "x2": 2.0},
        ensemble_disagreement=5.0,
        ensemble_disagreement_threshold=1.0,
    )

    assert low_disagreement.within_training_domain is True
    assert high_disagreement.within_training_domain is False
    assert any("disagree" in w for w in high_disagreement.warnings)


def test_no_numeric_features_still_produces_an_assessment() -> None:
    schema = DatasetSchema(features=["cat"], target="y", categorical_features=["cat"])
    x_train = pd.DataFrame({"cat": ["a", "b", "a"]})
    detector = fit_ood_detector(x_train, schema)

    assessment = assess_ood(detector, {"cat": "a"})

    assert np.isnan(assessment.nearest_neighbor_distance)
    assert assessment.nearest_neighbor_distance_threshold is None

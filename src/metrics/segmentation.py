from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import adjusted_rand_score
from skimage.metrics import variation_of_information



def confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[int, int, int, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = int(np.logical_and(prediction, target).sum())
    true_negative = int(np.logical_and(~prediction, ~target).sum())
    false_positive = int(np.logical_and(prediction, ~target).sum())
    false_negative = int(np.logical_and(~prediction, target).sum())

    return true_positive, true_negative, false_positive, false_negative


def binary_segmentation_metrics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    prediction = probability >= threshold
    target = target >= 0.5

    tp, tn, fp, fn = confusion_counts(prediction, target)
    epsilon = 1e-8

    dice = 2.0 * tp / (2.0 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def partition_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    prediction = prediction.astype(np.int32)
    target = target.astype(np.int32)

    ari = adjusted_rand_score(target.ravel(), prediction.ravel())
    split_vi, merge_vi = variation_of_information(target, prediction)

    return {
        "ari": float(ari),
        "voi": float(split_vi + merge_vi),
    }


def average_symmetric_surface_distance(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("inf")

    prediction_boundary = np.logical_xor(
        prediction,
        binary_erosion(prediction),
    )
    target_boundary = np.logical_xor(
        target,
        binary_erosion(target),
    )

    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_prediction = distance_transform_edt(~prediction_boundary)

    distances = np.concatenate(
        [
            distance_to_target[prediction_boundary],
            distance_to_prediction[target_boundary],
        ]
    )
    return float(distances.mean())

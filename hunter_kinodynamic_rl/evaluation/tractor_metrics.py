"""Architecture-independent prediction, ranking, calibration and CI metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _finite_pair(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    valid = np.isfinite(prediction) & np.isfinite(target)
    return prediction[valid], target[valid]


def binary_calibration_metrics(probability, outcome, bins: int = 10) -> dict:
    probability, outcome = _finite_pair(probability, outcome)
    if probability.size == 0:
        return {"count": 0, "event_count": 0, "brier": None, "nll": None, "ece": None}
    if np.any((probability < 0.0) | (probability > 1.0)) or np.any((outcome != 0) & (outcome != 1)):
        raise ValueError("binary calibration inputs are outside their domain")
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    brier = np.mean((probability - outcome) ** 2)
    nll = -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped))
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    reliability = []
    for index in range(bins):
        include = (probability >= boundaries[index]) & (
            probability < boundaries[index + 1] if index < bins - 1 else probability <= 1.0
        )
        count = int(include.sum())
        if count == 0:
            continue
        confidence = float(probability[include].mean())
        frequency = float(outcome[include].mean())
        ece += count / probability.size * abs(confidence - frequency)
        reliability.append({
            "lo": float(boundaries[index]), "hi": float(boundaries[index + 1]),
            "count": count, "mean_probability": confidence, "event_frequency": frequency,
        })
    return {
        "count": int(probability.size), "event_count": int(outcome.sum()),
        "brier": float(brier), "nll": float(nll), "ece": float(ece),
        "reliability": reliability,
    }


def candidate_ranking_metrics(score, utility, valid) -> dict:
    score = np.asarray(score, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if score.shape != utility.shape or score.shape != valid.shape or score.ndim != 2:
        raise ValueError("ranking tensors must be matching (N,K) arrays")
    regrets, ndcg, unsafe = [], [], []
    for row in range(score.shape[0]):
        indices = np.flatnonzero(valid[row] & np.isfinite(score[row]) & np.isfinite(utility[row]))
        if indices.size == 0:
            continue
        selected = indices[np.argmax(score[row, indices])]
        best_utility = float(np.max(utility[row, indices]))
        regrets.append(best_utility - float(utility[row, selected]))
        unsafe.append(float(utility[row, selected] < 0.0))
        predicted_order = indices[np.argsort(-score[row, indices], kind="stable")]
        ideal_order = indices[np.argsort(-utility[row, indices], kind="stable")]
        shifted = utility[row, indices] - np.min(utility[row, indices])
        gains = {int(index): float(value) for index, value in zip(indices, shifted)}
        discounts = 1.0 / np.log2(np.arange(indices.size) + 2.0)
        dcg = sum(gains[int(index)] * discounts[position] for position, index in enumerate(predicted_order))
        idcg = sum(gains[int(index)] * discounts[position] for position, index in enumerate(ideal_order))
        ndcg.append(1.0 if idcg <= 1e-12 else dcg / idcg)
    if not regrets:
        return {"row_count": 0, "mean_regret": None, "ndcg": None, "unsafe_top1_rate": None}
    return {
        "row_count": len(regrets), "mean_regret": float(np.mean(regrets)),
        "median_regret": float(np.median(regrets)), "ndcg": float(np.mean(ndcg)),
        "unsafe_top1_rate": float(np.mean(unsafe)), "unsafe_top1_count": int(np.sum(unsafe)),
    }


def paired_bootstrap_interval(
    differences: Sequence[float], confidence: float = 0.95,
    bootstrap_samples: int = 10000, seed: int = 0,
) -> dict:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "low": None, "high": None}
    if not 0.0 < confidence < 1.0 or bootstrap_samples <= 0:
        raise ValueError("invalid bootstrap interval settings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(bootstrap_samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "count": int(values.size), "mean": float(values.mean()),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "confidence": confidence, "bootstrap_samples": bootstrap_samples, "seed": seed,
    }


def occupancy_confusion(predicted_class, target_class, valid, classes: int = 4) -> dict:
    predicted = np.asarray(predicted_class)
    target = np.asarray(target_class)
    valid = np.asarray(valid, dtype=bool)
    if predicted.shape != target.shape or target.shape != valid.shape:
        raise ValueError("occupancy arrays must have matching shapes")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (target[valid].astype(int), predicted[valid].astype(int)), 1)
    rows = []
    for class_id in range(classes):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        rows.append({
            "class_id": class_id, "support": int(confusion[class_id].sum()),
            "iou": None if tp + fp + fn == 0 else float(tp / (tp + fp + fn)),
            "f1": None if 2 * tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn)),
        })
    return {"valid_cells": int(valid.sum()), "confusion": confusion.tolist(), "per_class": rows}

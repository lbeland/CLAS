"""Pooled confusion-matrix + error metrics for a set of (estimate, ground
truth) pairs."""
import numpy as np


def compute_pooled_metrics(estimates, ground_truths):
    """Confusion-matrix + error metrics pooled over many (estimate, gt)
    pairs -- e.g. every seed of one condition -- since rmse and f1_score
    are nonlinear and can't be recovered by averaging n=1 per-trial values
    afterwards. errors stays nan-padded to len(estimates) (not compacted)
    so it stays index-aligned with estimates for per-seed storage."""
    estimates = np.asarray(estimates, dtype=float)
    ground_truths = np.asarray(ground_truths, dtype=float)
    errors = np.full(estimates.shape, np.nan)
    fp = fn = tn = tp = 0

    for i, (est, gt) in enumerate(zip(estimates, ground_truths)):
        est_none, gt_none = np.isnan(est), np.isnan(gt)
        if gt_none and est_none:
            tn += 1
        elif gt_none and not est_none:
            fp += 1
            errors[i] = est
        elif not gt_none and est_none:
            fn += 1
        else:
            tp += 1
            errors[i] = est - gt

    abs_errors = np.abs(errors[~np.isnan(errors)])
    n = len(estimates)
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    accuracy = (tp + tn) / n if n > 0 else np.nan
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        "estimates":      estimates,
        "errors":         errors,
        "mae":            np.mean(abs_errors) if abs_errors.size else np.nan,
        "rmse":           np.sqrt(np.mean(abs_errors ** 2)) if abs_errors.size else np.nan,
        "std":            np.std(abs_errors) if abs_errors.size else np.nan,
        "fn":             fn,
        "fp":             fp,
        "n":              n,
        "fail_rate":      (fn + fp) / n,
        "accuracy":       accuracy,
        "precision":      precision,
        "recall":         recall,
        "f1_score":       f1_score,
    }

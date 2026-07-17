"""Pure NumPy helpers for leakage-free shared-bottleneck evaluation."""

import numpy as np


def _augment(features):
    return np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)


def balanced_accuracy(prediction, target):
    """Return the mean class-conditional accuracy for binary {-1, +1} targets."""
    scores = [
        float((prediction[target == value] == value).mean())
        for value in (-1.0, 1.0)
        if np.any(target == value)
    ]
    return float(np.mean(scores)) if scores else 0.5


def coefficient_of_determination(prediction, target):
    """Held-out R² = 1 - SSE/SST, retaining negative values when a probe fails."""
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - target.mean()) ** 2)
    return 0.0 if total <= 1e-12 else float(1.0 - residual / total)


def fit_whitener(features, rel_eig_threshold=1e-6):
    """Fit a PCA whitener and return (mean, projection) using these rows only."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    covariance = centered.T @ centered / max(features.shape[0], 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if eigenvalues.size == 0 or eigenvalues[0] <= 0:
        raise ValueError("Cannot whiten features with zero variance")
    keep = eigenvalues > rel_eig_threshold * eigenvalues[0]
    projection = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    return mean, projection


def shared_bottleneck_accuracy(
    features,
    targets,
    r_list,
    train_frac=0.6,
    seed=0,
    rel_eig_threshold=1e-6,
):
    """Evaluate a train-fitted shared bottleneck on a disjoint held-out split.

    Centering, whitening, bottleneck estimation, and task heads are all fit on
    training rows. ``mean_recov`` is genuine held-out R², not squared Pearson
    correlation, and may therefore be negative.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be two-dimensional")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must lie strictly between zero and one")

    n, n_tasks = targets.shape
    permutation = np.random.default_rng(seed).permutation(n)
    n_train = int(train_frac * n)
    if n_train < 2 or n - n_train < 2:
        raise ValueError("train and test splits must each contain at least two rows")
    train_idx, test_idx = permutation[:n_train], permutation[n_train:]
    x_train_raw, x_test_raw = features[train_idx], features[test_idx]
    y_train, y_test = targets[train_idx], targets[test_idx]

    mean, whitener = fit_whitener(x_train_raw, rel_eig_threshold)
    x_train = (x_train_raw - mean) @ whitener
    x_test = (x_test_raw - mean) @ whitener

    normalized_targets = (y_train - y_train.mean(0)) / (y_train.std(0) + 1e-12)
    task_vectors = x_train.T @ normalized_targets / x_train.shape[0]
    task_moment = task_vectors @ task_vectors.T / n_tasks
    eigenvalues, eigenvectors = np.linalg.eigh(task_moment)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, directions = eigenvalues[order], eigenvectors[:, order]

    mean_balanced, per_task_balanced, mean_recovery = [], [], []
    for rank in r_list:
        rank = int(rank)
        if rank < 1:
            raise ValueError("Every bottleneck rank must be positive")
        projection = directions[:, : min(rank, directions.shape[1])]
        train_design = _augment(x_train @ projection)
        test_design = _augment(x_test @ projection)
        balanced, recovery = [], []
        for task in range(n_tasks):
            weights, *_ = np.linalg.lstsq(train_design, y_train[:, task], rcond=None)
            prediction = test_design @ weights
            signed_prediction = np.where(prediction >= 0, 1.0, -1.0)
            balanced.append(balanced_accuracy(signed_prediction, y_test[:, task]))
            recovery.append(
                coefficient_of_determination(prediction, y_test[:, task])
            )
        per_task_balanced.append(balanced)
        mean_balanced.append(float(np.mean(balanced)))
        mean_recovery.append(float(np.mean(recovery)))

    return {
        "mean_bal_acc": np.asarray(mean_balanced),
        "per_task_bal_acc": np.asarray(per_task_balanced),
        "mean_recov": np.asarray(mean_recovery),
        "eigvals": eigenvalues,
        "whitening_rank": int(whitener.shape[1]),
        "whitening_scope": "train_split_only",
    }


def evaluate_bottleneck_splits(
    features,
    targets,
    r_list,
    split_seeds,
    train_frac=0.6,
    rel_eig_threshold=1e-6,
):
    """Aggregate the leakage-free evaluation across independent data splits."""
    seeds = [int(seed) for seed in split_seeds]
    if not seeds:
        raise ValueError("At least one split seed is required")
    results = [
        shared_bottleneck_accuracy(
            features,
            targets,
            r_list,
            train_frac=train_frac,
            seed=seed,
            rel_eig_threshold=rel_eig_threshold,
        )
        for seed in seeds
    ]

    def summarize(key):
        values = np.stack([result[key] for result in results])
        return values.mean(axis=0), values.std(axis=0, ddof=0)

    mean_bal, mean_bal_sd = summarize("mean_bal_acc")
    per_task, per_task_sd = summarize("per_task_bal_acc")
    recovery, recovery_sd = summarize("mean_recov")
    capacities = []
    for result in results:
        eigenvalues = np.clip(result["eigvals"], 0.0, None)
        total = eigenvalues.sum()
        if total <= 0:
            raise ValueError("Task moment has zero energy")
        cumulative = np.cumsum(eigenvalues) / total
        capacities.append([
            cumulative[min(int(rank), cumulative.size) - 1] for rank in r_list
        ])
    capacities = np.asarray(capacities)

    return {
        "mean_bal_acc": mean_bal,
        "mean_bal_acc_sd": mean_bal_sd,
        "per_task_bal_acc": per_task,
        "per_task_bal_acc_sd": per_task_sd,
        "mean_recov": recovery,
        "mean_recov_sd": recovery_sd,
        "capacity": capacities.mean(axis=0),
        "capacity_sd": capacities.std(axis=0, ddof=0),
        "whitening_ranks": [result["whitening_rank"] for result in results],
        "whitening_scope": "train_split_only",
        "split_seeds": seeds,
    }

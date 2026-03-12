import random
from typing import Callable, List, Any, Sequence, Tuple
import numpy as np

def bootstrap_mean(values: Sequence[float], n_iter: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    """Compute mean and 95% CI via bootstrap for a list of numeric values.

    Returns (mean, low, high)
    """
    rng = random.Random(seed)
    vals = list(values)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_iter):
        sample = [rng.choice(vals) for _ in range(n)]
        means.append(float(np.mean(sample)))
    means = sorted(means)
    low = means[int(0.025 * len(means))]
    high = means[int(0.975 * len(means))]
    return float(np.mean(vals)), float(low), float(high)

def bootstrap_metric_per_item(per_item_scores: Sequence[float], n_iter: int = 1000, seed: int = 0):
    return bootstrap_mean(per_item_scores, n_iter=n_iter, seed=seed)

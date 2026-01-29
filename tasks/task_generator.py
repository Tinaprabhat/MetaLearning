# tasks/task_generator.py

import numpy as np
from .classification_tasks import generate_classification_task


def generate_task(
    n_samples=200,
    noise=0.1,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    return generate_classification_task(
        n_samples=n_samples,
        noise=noise
    )


def generate_shifted_task(
    base_seed,
    shift_scale=4.0,
    n_samples=200,
    noise = 0.25
):
    """
    Generates a task from a shifted input distribution.
    """
    np.random.seed(base_seed)

    X, y = generate_classification_task(
        n_samples=n_samples,
        noise=noise
    )

    shift = np.random.uniform(
        -shift_scale, shift_scale, size=2
    )

    X = X + shift
    return X, y

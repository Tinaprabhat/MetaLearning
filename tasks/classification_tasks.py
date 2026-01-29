# tasks/classification_tasks.py

import numpy as np

def generate_classification_task(n_samples=200, noise=0.1):
    """
    Generates a binary classification task with random class centers.
    """

    # Randomly sample class centers (distribution shift)
    center_1 = np.random.uniform(-5, 5, size=2)
    center_2 = np.random.uniform(-5, 5, size=2)

    X1 = np.random.randn(n_samples // 2, 2) + center_1
    X2 = np.random.randn(n_samples // 2, 2) + center_2

    X = np.vstack([X1, X2])
    y = np.hstack([
        np.zeros(n_samples // 2),
        np.ones(n_samples // 2)
    ])

    # Add noise
    X += noise * np.random.randn(*X.shape)

    return X.astype(np.float32), y.astype(np.int64)

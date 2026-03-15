# tests/test_task_generator.py
"""
Tests for tasks/task_generator.py and tasks/classification_tasks.py.

Run with:
    pytest tests/test_task_generator.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.task_generator import generate_task, generate_shifted_task
from tasks.classification_tasks import generate_classification_task


# ---------------------------------------------------------------------------
# generate_classification_task
# ---------------------------------------------------------------------------

class TestGenerateClassificationTask:

    def test_output_shapes(self):
        X, y = generate_classification_task(n_samples=200)
        assert X.shape == (200, 2), f"Expected X shape (200, 2), got {X.shape}"
        assert y.shape == (200,),   f"Expected y shape (200,), got {y.shape}"

    def test_dtypes(self):
        X, y = generate_classification_task(n_samples=100)
        assert X.dtype == np.float32, f"Expected float32, got {X.dtype}"
        assert y.dtype == np.int64,   f"Expected int64, got {y.dtype}"

    def test_binary_labels(self):
        _, y = generate_classification_task(n_samples=200)
        unique = set(y.tolist())
        assert unique == {0, 1}, f"Expected labels {{0, 1}}, got {unique}"

    def test_balanced_classes(self):
        _, y = generate_classification_task(n_samples=200)
        assert y.sum() == 100, f"Expected 100 positives, got {int(y.sum())}"

    def test_different_seeds_produce_different_data(self):
        # classification_tasks uses global numpy state — call generate_task
        # with explicit seeds for reproducibility checks instead
        np.random.seed(0)
        X1, _ = generate_classification_task()
        np.random.seed(1)
        X2, _ = generate_classification_task()
        assert not np.allclose(X1, X2), "Different seeds should produce different data"

    def test_noise_param_affects_spread(self):
        np.random.seed(42)
        X_clean, _ = generate_classification_task(n_samples=500, noise=0.0)
        np.random.seed(42)
        X_noisy, _ = generate_classification_task(n_samples=500, noise=5.0)
        assert X_noisy.std() > X_clean.std(), "Higher noise should increase spread"

    @pytest.mark.parametrize("n_samples", [20, 100, 500])
    def test_various_sample_sizes(self, n_samples):
        X, y = generate_classification_task(n_samples=n_samples)
        assert len(X) == n_samples
        assert len(y) == n_samples


# ---------------------------------------------------------------------------
# generate_task (seeded wrapper)
# ---------------------------------------------------------------------------

class TestGenerateTask:

    def test_same_seed_same_data(self):
        X1, y1 = generate_task(seed=42)
        X2, y2 = generate_task(seed=42)
        assert np.array_equal(X1, X2), "Same seed should produce identical X"
        assert np.array_equal(y1, y2), "Same seed should produce identical y"

    def test_different_seeds_different_data(self):
        X1, _ = generate_task(seed=42)
        X2, _ = generate_task(seed=43)
        assert not np.array_equal(X1, X2), "Different seeds should produce different data"

    def test_output_shape_default(self):
        X, y = generate_task(seed=0)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_output_shape_custom(self):
        X, y = generate_task(n_samples=80, seed=7)
        assert X.shape == (80, 2)
        assert y.shape == (80,)

    def test_no_seed_runs_without_error(self):
        X, y = generate_task()
        assert X.shape[1] == 2

    def test_multiple_tasks_are_distinct(self):
        """Generate a batch of tasks and assert no two are identical."""
        n_tasks = 10
        tasks = [generate_task(seed=i) for i in range(n_tasks)]
        for i in range(n_tasks):
            for j in range(i + 1, n_tasks):
                assert not np.array_equal(tasks[i][0], tasks[j][0]), (
                    f"Tasks {i} and {j} produced identical data — seed isolation broken"
                )

    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 256, 1000])
    def test_seeds_are_reproducible(self, seed):
        X1, y1 = generate_task(seed=seed)
        X2, y2 = generate_task(seed=seed)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_task_batch_covers_distribution(self):
        """
        With enough tasks, class centres should span a wide area.
        This catches degenerate cases where all tasks cluster in one region.
        """
        all_X = np.vstack([generate_task(seed=i)[0] for i in range(50)])
        assert all_X[:, 0].std() > 1.0, "Tasks should span diverse x-ranges"
        assert all_X[:, 1].std() > 1.0, "Tasks should span diverse y-ranges"


# ---------------------------------------------------------------------------
# generate_shifted_task
# ---------------------------------------------------------------------------

class TestGenerateShiftedTask:

    def test_output_shapes(self):
        X, y = generate_shifted_task(base_seed=42)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_dtypes(self):
        X, y = generate_shifted_task(base_seed=0)
        assert X.dtype == np.float32
        assert y.dtype == np.int64

    def test_binary_labels(self):
        _, y = generate_shifted_task(base_seed=5)
        assert set(y.tolist()) == {0, 1}

    def test_same_seed_reproducible(self):
        X1, y1 = generate_shifted_task(base_seed=7)
        X2, y2 = generate_shifted_task(base_seed=7)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)

    def test_different_seeds_different_shift(self):
        X1, _ = generate_shifted_task(base_seed=10)
        X2, _ = generate_shifted_task(base_seed=11)
        assert not np.array_equal(X1, X2)

    def test_shift_moves_data_away_from_base(self):
        """
        With a large shift_scale the shifted task's mean should be far
        from the un-shifted task's mean.
        """
        X_base, _ = generate_task(seed=42)
        X_shifted, _ = generate_shifted_task(base_seed=42, shift_scale=10.0)
        base_mean = X_base.mean(axis=0)
        shift_mean = X_shifted.mean(axis=0)
        dist = np.linalg.norm(shift_mean - base_mean)
        assert dist > 1.0, (
            f"Large shift_scale should move data significantly; distance was {dist:.3f}"
        )

    def test_zero_shift_scale_matches_base(self):
        """shift_scale=0 should produce a task identical to generate_task."""
        X_base, y_base = generate_task(seed=42)
        X_shift, y_shift = generate_shifted_task(base_seed=42, shift_scale=0.0)
        assert np.allclose(X_base, X_shift, atol=1e-5), (
            "Zero shift should not move data"
        )
        assert np.array_equal(y_base, y_shift)

    @pytest.mark.parametrize("shift_scale", [0.5, 2.0, 4.0, 8.0])
    def test_shift_scale_param(self, shift_scale):
        X, y = generate_shifted_task(base_seed=0, shift_scale=shift_scale)
        assert X.shape == (200, 2)
        assert y.shape == (200,)

    def test_multiple_shifted_tasks_distinct(self):
        tasks = [generate_shifted_task(base_seed=i) for i in range(10)]
        for i in range(10):
            for j in range(i + 1, 10):
                assert not np.array_equal(tasks[i][0], tasks[j][0]), (
                    f"Shifted tasks {i} and {j} should be distinct"
                )

    def test_shifted_tasks_span_diverse_range(self):
        """Shifted tasks should cover a wider input range than normal tasks."""
        normal_X = np.vstack([generate_task(seed=i)[0] for i in range(20)])
        shifted_X = np.vstack([generate_shifted_task(base_seed=i)[0] for i in range(20)])
        assert shifted_X.std() >= normal_X.std() * 0.8, (
            "Shifted tasks should have comparable or wider spread than normal tasks"
        )


# ---------------------------------------------------------------------------
# Cross-generator consistency
# ---------------------------------------------------------------------------

class TestCrossGeneratorConsistency:

    def test_task_and_shifted_share_label_structure(self):
        """Both generators should always produce exactly 50% class-1 labels."""
        for seed in range(10):
            _, y_n = generate_task(seed=seed)
            _, y_s = generate_shifted_task(base_seed=seed)
            assert y_n.mean() == 0.5, f"Normal task {seed}: unbalanced labels"
            assert y_s.mean() == 0.5, f"Shifted task {seed}: unbalanced labels"

    def test_generate_task_batch(self):
        """Utility: generate N tasks and verify they form a valid meta-learning batch."""
        n = 8
        tasks = [(generate_task(seed=i), generate_shifted_task(base_seed=i + 100))
                 for i in range(n)]

        assert len(tasks) == n
        for i, ((X_n, y_n), (X_s, y_s)) in enumerate(tasks):
            assert X_n.shape == (200, 2), f"Task {i} normal X shape wrong"
            assert X_s.shape == (200, 2), f"Task {i} shifted X shape wrong"
            # Normal and shifted versions of the same seed should differ
            assert not np.array_equal(X_n, X_s), f"Task {i}: normal and shifted are identical"

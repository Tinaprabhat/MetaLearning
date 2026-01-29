import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# experiments/evaluate_task_shift.py

from experiments.train_base_learner import train_base_model
from experiments.train_with_meta_learner import train_with_meta


def evaluate_task_shift(
    train_seed=42,
    test_seed=99,
    loss_threshold=0.62
):
    print("\n--- Training META on Task A ---")
    meta_train = train_with_meta(
        seed=train_seed,
        shifted=False
    )
    trained_meta = meta_train["meta_model"]

    print("\n--- Evaluating on SHIFTED Task B ---")

    baseline_test = train_base_model(
        seed=test_seed,
        shifted=True,
        loss_threshold=loss_threshold
    )

    meta_test = train_with_meta(
        seed=test_seed,
        shifted=True,
        freeze_meta=True,
        meta_model=trained_meta,
        loss_threshold=loss_threshold
    )

    print("\nRESULTS (Task B):")
    print("Baseline threshold epoch:", baseline_test["threshold_epoch"])
    print("Meta threshold epoch:    ", meta_test["threshold_epoch"])


if __name__ == "__main__":
    evaluate_task_shift()

import matplotlib.pyplot as plt


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.train_base_learner import train_base_model
from experiments.train_with_meta_learner import train_with_meta

def evaluate():
    baseline = train_base_model()
    meta = train_with_meta()

    epochs = range(len(baseline["loss"]))

    # ---- LOSS COMPARISON ----
    plt.figure()
    plt.plot(epochs, baseline["loss"], label="Baseline SGD")
    plt.plot(epochs, meta["loss"], label="Meta-Optimized")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Convergence Comparison")
    plt.savefig("plots/loss_comparison.png")

    # ---- GRADIENT NORM COMPARISON ----
    plt.figure()
    plt.plot(epochs, baseline["grad_norm"], label="Baseline SGD")
    plt.plot(epochs, meta["grad_norm"], label="Meta-Optimized")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.title("Gradient Norm Dynamics")
    plt.savefig("plots/gradnorm_comparison.png")

    print("Evaluation completed. Plots saved in /plots.")

if __name__ == "__main__":
    evaluate()

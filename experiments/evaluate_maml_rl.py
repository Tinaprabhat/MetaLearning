# experiments/evaluate_maml_rl.py
"""
Evaluation: Baseline SGD  vs  MAML  vs  MAML + PPO lr-controller

Tests all three approaches on:
  (a) Same-distribution tasks  (in-distribution generalisation)
  (b) Shifted tasks             (out-of-distribution generalisation)

Outputs
-------
  plots/eval_loss_curves.png   — loss-vs-epoch for all three methods
  plots/eval_summary.png       — bar chart of final losses + epochs-to-threshold
  Console summary table
"""

import sys
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base_learner import BaseLearner
from tasks.task_generator import generate_task, generate_shifted_task
from meta_learners.maml_trainer import MAMLTrainer, _clone_params
from meta_learners.meta_env import MetaLearningEnv
from meta_learners.ppo_agent import PPOAgent
from torch.func import functional_call
from experiments.training_monitor import compute_gradient_norm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EVAL_CFG = {
    "epochs": 50,
    "base_lr": 0.01,
    # Threshold set low enough that all methods can cross it.
    # MAML starts at ~0.30 on shifted tasks, so 0.45 was never reachable
    # from below — it was always already below that value at epoch 0.
    "loss_threshold": 0.22,
    "n_eval_tasks": 5,
    "maml_inner_steps": 5,
    "maml_inner_lr": 0.05,
    "ckpt_maml": "checkpoints/maml_model.pt",
    "ckpt_ppo": "checkpoints/ppo_agent.pt",
    "plot_dir": "plots",
}


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def eval_baseline(X, y, cfg) -> dict:
    """Standard SGD from random init."""
    model = BaseLearner()
    optimizer = optim.SGD(model.parameters(), lr=cfg["base_lr"])
    criterion = nn.CrossEntropyLoss()

    losses, grad_norms = [], []
    threshold_epoch = None

    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        gn = compute_gradient_norm(model)
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(gn)
        if threshold_epoch is None and loss.item() <= cfg["loss_threshold"]:
            threshold_epoch = epoch

    return {"loss": losses, "grad_norm": grad_norms, "threshold_epoch": threshold_epoch}


def eval_maml(X, y, maml_model_path: str, cfg) -> dict:
    """MAML fast-adaptation then fine-tune."""
    criterion = nn.CrossEntropyLoss()

    # Load meta-parameters
    meta_model = BaseLearner()
    if Path(maml_model_path).exists():
        meta_model.load_state_dict(
            torch.load(maml_model_path, weights_only=True)
        )
    else:
        print(f"  [warn] MAML checkpoint not found at {maml_model_path}. "
              "Using random init — run train_maml_rl.py first.")

    maml = MAMLTrainer(
        model=meta_model,
        inner_lr=cfg["maml_inner_lr"],
        inner_steps=cfg["maml_inner_steps"],
    )

    # Fast-adapt to this task's support set (80 %)
    split = int(0.8 * len(X))
    adapted_params = maml.adapt(X[:split], y[:split])

    # Fine-tune adapted params with SGD on full data
    # We materialise adapted params into a new model for clean training
    fine_model = BaseLearner()
    with torch.no_grad():
        for name, param in fine_model.named_parameters():
            if name in adapted_params:
                param.copy_(adapted_params[name])

    optimizer = optim.SGD(fine_model.parameters(), lr=cfg["base_lr"])
    losses, grad_norms = [], []
    threshold_epoch = None

    for epoch in range(cfg["epochs"]):
        optimizer.zero_grad()
        loss = criterion(fine_model(X), y)
        loss.backward()
        gn = compute_gradient_norm(fine_model)
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(gn)
        if threshold_epoch is None and loss.item() <= cfg["loss_threshold"]:
            threshold_epoch = epoch

    return {"loss": losses, "grad_norm": grad_norms, "threshold_epoch": threshold_epoch}


def eval_maml_ppo(X, y, maml_model_path: str, ppo_ckpt_path: str, cfg) -> dict:
    """MAML fast-adaptation + PPO-controlled lr fine-tuning."""
    criterion = nn.CrossEntropyLoss()

    # Load meta-parameters
    meta_model = BaseLearner()
    if Path(maml_model_path).exists():
        meta_model.load_state_dict(
            torch.load(maml_model_path, weights_only=True)
        )

    maml = MAMLTrainer(
        model=meta_model,
        inner_lr=cfg["maml_inner_lr"],
        inner_steps=cfg["maml_inner_steps"],
    )

    split = int(0.8 * len(X))
    adapted_params = maml.adapt(X[:split], y[:split])

    # Materialise adapted params into a model we can hand to the env
    fine_model = BaseLearner()
    with torch.no_grad():
        for name, param in fine_model.named_parameters():
            if name in adapted_params:
                param.copy_(adapted_params[name])

    # Snapshot the adapted state so reset() can restore it each episode
    adapted_state = copy.deepcopy(fine_model.state_dict())

    # Load PPO agent
    ppo = PPOAgent(obs_dim=MetaLearningEnv.OBS_DIM)
    if Path(ppo_ckpt_path).exists():
        ppo.load(ppo_ckpt_path)
    else:
        print(f"  [warn] PPO checkpoint not found at {ppo_ckpt_path}. "
              "Using untrained agent — run train_maml_rl.py first.")

    env = MetaLearningEnv(
        model=fine_model,
        X=X,
        y=y,
        base_lr=cfg["base_lr"],
        max_steps=cfg["epochs"],
    )
    # Pass adapted_state so reset() loads MAML θ instead of re-doing xavier
    obs = env.reset(init_state_dict=adapted_state)

    losses, grad_norms, lr_history = [], [], []
    threshold_epoch = None

    for epoch in range(cfg["epochs"]):
        action, _, _ = ppo.select_action(obs)
        obs, reward, done = env.step(action)

        # Re-compute current loss for logging (env tracks model internally)
        with torch.no_grad():
            logits = fine_model(X)
            loss = criterion(logits, y).item()
        gn = compute_gradient_norm(fine_model)

        losses.append(loss)
        grad_norms.append(gn)
        lr_history.append(cfg["base_lr"] * float(action))

        if threshold_epoch is None and loss <= cfg["loss_threshold"]:
            threshold_epoch = epoch

        if done:
            break

    return {
        "loss": losses,
        "grad_norm": grad_norms,
        "lr": lr_history,
        "threshold_epoch": threshold_epoch,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_loss_curves(results_normal, results_shifted, cfg):
    Path(cfg["plot_dir"]).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ["Baseline SGD", "MAML", "MAML + PPO"]
    keys = ["baseline", "maml", "maml_ppo"]
    colors = ["steelblue", "darkorange", "seagreen"]

    for ax, (results, title) in zip(
        axes,
        [(results_normal, "In-distribution tasks"),
         (results_shifted, "Shifted tasks (OOD)")],
    ):
        for key, label, color in zip(keys, labels, colors):
            if key not in results:
                continue
            losses = results[key]["loss"]
            # Average across tasks
            avg = np.mean(losses, axis=0) if np.ndim(losses) > 1 else losses
            ax.plot(avg, label=label, color=color)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg["plot_dir"], "eval_loss_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Loss curves saved -> {out}")


def print_summary_table(results_normal, results_shifted):
    labels = {"baseline": "Baseline SGD", "maml": "MAML", "maml_ppo": "MAML + PPO"}
    header = f"{'Method':<20} {'Normal final loss':>18} {'Shifted final loss':>18} "
    header += f"{'Normal thresh ep':>17} {'Shifted thresh ep':>17}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for key, label in labels.items():
        n_loss = (
            f"{np.mean([r['loss'][-1] for r in results_normal[key]]):.4f}"
            if key in results_normal else "N/A"
        )
        s_loss = (
            f"{np.mean([r['loss'][-1] for r in results_shifted[key]]):.4f}"
            if key in results_shifted else "N/A"
        )

        def _avg_thresh(res_list):
            vals = [r['threshold_epoch'] for r in res_list]
            reached = [v for v in vals if v is not None]
            if not reached:
                return "never"
            if len(reached) < len(vals):
                return f"{np.mean(reached):.1f} ({len(reached)}/{len(vals)} tasks)"
            return f"{np.mean(reached):.1f}"

        n_thresh = _avg_thresh(results_normal[key]) if key in results_normal else "N/A"
        s_thresh = _avg_thresh(results_shifted[key]) if key in results_shifted else "N/A"

        print(f"{label:<20} {n_loss:>18} {s_loss:>18} {n_thresh:>17} {s_thresh:>17}")
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = EVAL_CFG

    results_normal = {"baseline": [], "maml": [], "maml_ppo": []}
    results_shifted = {"baseline": [], "maml": [], "maml_ppo": []}

    for task_seed in range(cfg["n_eval_tasks"]):
        # ---- In-distribution ----
        X_n, y_n = generate_task(seed=task_seed + 200)
        X_n = torch.tensor(X_n, dtype=torch.float32)
        y_n = torch.tensor(y_n, dtype=torch.long)

        print(f"\nTask {task_seed+1}/{cfg['n_eval_tasks']} (normal) …")
        results_normal["baseline"].append(eval_baseline(X_n, y_n, cfg))
        results_normal["maml"].append(eval_maml(X_n, y_n, cfg["ckpt_maml"], cfg))
        results_normal["maml_ppo"].append(
            eval_maml_ppo(X_n, y_n, cfg["ckpt_maml"], cfg["ckpt_ppo"], cfg)
        )

        # ---- Shifted ----
        X_s, y_s = generate_shifted_task(base_seed=task_seed + 200)
        X_s = torch.tensor(X_s, dtype=torch.float32)
        y_s = torch.tensor(y_s, dtype=torch.long)

        print(f"Task {task_seed+1}/{cfg['n_eval_tasks']} (shifted) …")
        results_shifted["baseline"].append(eval_baseline(X_s, y_s, cfg))
        results_shifted["maml"].append(eval_maml(X_s, y_s, cfg["ckpt_maml"], cfg))
        results_shifted["maml_ppo"].append(
            eval_maml_ppo(X_s, y_s, cfg["ckpt_maml"], cfg["ckpt_ppo"], cfg)
        )

    # Aggregate for plotting (stack losses per method)
    def _stack(res_dict, key):
        return {
            "loss": [r["loss"] for r in res_dict[key]],
            "threshold_epoch": [r["threshold_epoch"] for r in res_dict[key]],
        }

    plot_data_normal = {k: _stack(results_normal, k) for k in results_normal}
    plot_data_shifted = {k: _stack(results_shifted, k) for k in results_shifted}
    plot_loss_curves(plot_data_normal, plot_data_shifted, cfg)
    print_summary_table(results_normal, results_shifted)


if __name__ == "__main__":
    main()
# experiments/train_maml_rl.py
"""
Unified MAML + PPO training loop.

Two-phase training
------------------
Phase 1 — MAML pre-training
    Trains the base-learner's meta-parameters θ over a distribution of tasks
    so that a few inner-loop gradient steps produce a well-adapted model.

Phase 2 — RL fine-tuning
    Freezes (or continues updating) θ and trains a PPO agent to adaptively
    control the learning rate during each training episode.  The agent
    observes [loss, grad_norm, weight_norm] and outputs a lr scale ∈ (0, 1).

The two phases share the same task generator, ensuring the RL agent
encounters the same distribution MAML was trained on.
"""

import sys
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.base_learner import BaseLearner
from tasks.task_generator import generate_task, generate_shifted_task
from meta_learners.maml_trainer import MAMLTrainer
from meta_learners.meta_env import MetaLearningEnv
from meta_learners.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = {
    # MAML
    "maml_meta_epochs": 100,     # outer-loop iterations
    "maml_meta_batch": 4,        # tasks per outer update
    "maml_inner_steps": 5,       # inner-loop gradient steps
    "maml_inner_lr": 0.05,
    "maml_outer_lr": 1e-3,
    # RL (PPO)
    "rl_episodes": 200,          # total training episodes
    "rl_max_steps": 50,          # env steps per episode (= epochs)
    "rl_base_lr": 0.01,
    # Shared
    "n_tasks": 8,                # tasks per MAML meta-batch pool
    "seed": 42,
    # Output
    "save_dir": "checkpoints",
    "plot_dir": "plots",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tasks(n: int, shifted: bool = False) -> list:
    """Return n (X_s, y_s, X_q, y_q) tuples as float tensors."""
    tasks = []
    for seed in range(n):
        if shifted:
            X, y = generate_shifted_task(base_seed=seed + 100)
        else:
            X, y = generate_task(seed=seed)

        # Split into support (80 %) and query (20 %)
        split = int(0.8 * len(X))
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        tasks.append((X_t[:split], y_t[:split], X_t[split:], y_t[split:]))
    return tasks


def _eval_adapted_loss(maml: MAMLTrainer, tasks: list) -> float:
    """Average query-set loss after fast adaptation — used for MAML logging."""
    total = 0.0
    for X_s, y_s, X_q, y_q in tasks:
        total += maml.evaluate(X_s, y_s, X_q, y_q)
    return total / len(tasks)


# ---------------------------------------------------------------------------
# Phase 1: MAML pre-training
# ---------------------------------------------------------------------------

def run_maml_phase(cfg: dict) -> tuple[BaseLearner, list]:
    print("\n" + "=" * 60)
    print("Phase 1 — MAML meta-training")
    print("=" * 60)

    torch.manual_seed(cfg["seed"])
    model = BaseLearner()
    maml = MAMLTrainer(
        model=model,
        inner_lr=cfg["maml_inner_lr"],
        outer_lr=cfg["maml_outer_lr"],
        inner_steps=cfg["maml_inner_steps"],
        meta_batch=cfg["maml_meta_batch"],
    )

    meta_losses = []
    for epoch in range(cfg["maml_meta_epochs"]):
        tasks = _make_tasks(cfg["n_tasks"])
        loss = maml.meta_train_step(tasks)
        meta_losses.append(loss)

        if (epoch + 1) % 10 == 0:
            eval_loss = _eval_adapted_loss(maml, tasks)
            print(f"  Epoch {epoch+1:4d}/{cfg['maml_meta_epochs']} "
                  f"| meta-loss: {loss:.4f}  eval-loss: {eval_loss:.4f}")

    # Save checkpoint
    Path(cfg["save_dir"]).mkdir(exist_ok=True)
    ckpt = os.path.join(cfg["save_dir"], "maml_model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"\nMAML checkpoint saved -> {ckpt}")

    return model, meta_losses


# ---------------------------------------------------------------------------
# Phase 2: RL (PPO) training
# ---------------------------------------------------------------------------

def run_rl_phase(
    meta_model: BaseLearner,
    cfg: dict,
) -> tuple[PPOAgent, list]:
    print("\n" + "=" * 60)
    print("Phase 2 — PPO lr-controller training")
    print("=" * 60)

    ppo = PPOAgent(obs_dim=MetaLearningEnv.OBS_DIM)

    episode_returns = []
    policy_losses = []

    for episode in range(cfg["rl_episodes"]):
        # Fresh task and a copy of the MAML-initialised weights for this episode
        seed = episode % cfg["n_tasks"]
        X, y = generate_task(seed=seed)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        episode_model = BaseLearner()

        env = MetaLearningEnv(
            model=episode_model,
            X=X_t,
            y=y_t,
            base_lr=cfg["rl_base_lr"],
            max_steps=cfg["rl_max_steps"],
        )

        # Pass the MAML state dict so reset() restores θ instead of random init
        obs = env.reset(init_state_dict=copy.deepcopy(meta_model.state_dict()))
        ep_return = 0.0

        for _ in range(cfg["rl_max_steps"]):
            action, log_prob, value = ppo.select_action(obs)
            next_obs, reward, done = env.step(action)

            ppo.buffer.add(obs, action, log_prob, reward, value, done)
            obs = next_obs
            ep_return += reward

            if done:
                break

        # PPO update after each episode
        stats = ppo.update()
        episode_returns.append(ep_return)
        policy_losses.append(stats["policy_loss"])

        if (episode + 1) % 20 == 0:
            avg_ret = np.mean(episode_returns[-20:])
            print(f"  Episode {episode+1:4d}/{cfg['rl_episodes']} "
                  f"| avg-return (last 20): {avg_ret:+.4f} "
                  f"| policy-loss: {stats['policy_loss']:.4f} "
                  f"| entropy: {stats['entropy']:.4f}")

    # Save PPO checkpoint
    ckpt = os.path.join(cfg["save_dir"], "ppo_agent.pt")
    ppo.save(ckpt)
    print(f"\nPPO checkpoint saved -> {ckpt}")

    return ppo, episode_returns


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_training_plots(
    meta_losses: list,
    episode_returns: list,
    cfg: dict,
):
    Path(cfg["plot_dir"]).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(meta_losses)
    axes[0].set_title("MAML meta-loss")
    axes[0].set_xlabel("Outer epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    # Smooth returns with a rolling window
    window = 10
    smoothed = np.convolve(
        episode_returns, np.ones(window) / window, mode="valid"
    )
    axes[1].plot(episode_returns, alpha=0.3, label="raw")
    axes[1].plot(
        range(window - 1, len(episode_returns)),
        smoothed,
        label=f"{window}-ep avg",
    )
    axes[1].set_title("PPO episode return")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg["plot_dir"], "maml_rl_training.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Training plots saved -> {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Phase 1
    meta_model, meta_losses = run_maml_phase(CFG)

    # Phase 2 — RL on top of MAML initialisation
    ppo_agent, episode_returns = run_rl_phase(meta_model, CFG)

    # Plots
    save_training_plots(meta_losses, episode_returns, CFG)

    print("\nAll done.  Run evaluate_maml_rl.py to benchmark on shifted tasks.")


if __name__ == "__main__":
    main()
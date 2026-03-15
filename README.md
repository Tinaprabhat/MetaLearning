# Meta-Learning with MAML + PPO

A PyTorch framework that combines **Model-Agnostic Meta-Learning (MAML)** with a **PPO reinforcement-learning agent** for adaptive learning-rate control.  The goal: a model that learns *how to learn* across a distribution of tasks and adapts its own optimisation strategy via RL.

---

## Architecture overview

```
Task distribution (T₁ … Tₙ)
        │
        ▼
┌──────────────────────────────────┐
│        MAML outer loop           │   ← Phase 1
│  ┌──────────────┐  ┌───────────┐ │
│  │  Fast adapt  │→ │ Query loss│ │
│  │  k-step SGD  │  │  ∇θ meta  │ │
│  └──────────────┘  └───────────┘ │
│         └────── meta update θ ───┘
└──────────────────────────────────┘
        │  (θ = meta-parameters)
        ▼
┌──────────────────────────────────┐
│     PPO lr-controller            │   ← Phase 2
│  state:  [loss, ‖∇‖, ‖w‖]       │
│  action: lr_scale ∈ (0, 1)       │
│  reward: Δloss improvement       │
└──────────────────────────────────┘
        │
        ▼
   Base learner (MLP)  ← adapted from θ, lr from PPO
```

---

## Project structure

```
MetaLearning/
├── models/
│   └── base_learner.py          # 2→32→32→2 MLP (the student)
│
├── tasks/
│   ├── classification_tasks.py  # 2-D binary classification generator
│   └── task_generator.py        # seed control + shifted task variant
│
├── meta_learners/
│   ├── maml_trainer.py          # MAML inner/outer loop (NEW)
│   ├── ppo_agent.py             # PPO actor-critic agent (NEW)
│   ├── meta_env.py              # RL env wrapping one training run (NEW)
│   
│
└── experiments/
    ├── train_maml_rl.py         # Phase 1 + Phase 2 training (NEW)
    ├── evaluate_maml_rl.py      # 3-way comparison + plots (NEW)
    ├── train_base_learner.py    # original baseline (kept)
    ├── test_task_generator.py
    └── training_monitor.py
```

---

## Quick start

### 1. Install dependencies

```bash
pip install torch numpy matplotlib
```

### 2. Train (MAML + PPO)

```bash
cd MetaLearning
python -m experiments.train_maml_rl
```

This runs two phases:
- **Phase 1**: 100 MAML outer epochs over 8 tasks
- **Phase 2**: 200 PPO episodes, each a 50-epoch training run

Checkpoints are saved to `checkpoints/`.

### 3. Evaluate

```bash
python -m experiments.evaluate_maml_rl
```

Compares **Baseline SGD**, **MAML**, and **MAML + PPO** on both normal and distribution-shifted tasks.  Plots are saved to `plots/`.

---

## How it works

### MAML (Phase 1)

MAML learns initial parameters θ that are *close to good solutions* for all tasks in the distribution.  For each task T:

1. **Inner loop** — compute adapted parameters φ = θ − α ∇_θ L_T(θ) (k steps)
2. **Outer loop** — update θ to minimise query-set loss using adapted φ

After training, a brand-new task only needs a few gradient steps to achieve low loss.

### PPO lr-controller (Phase 2)

A Proximal Policy Optimisation agent observes training diagnostics and outputs a learning-rate scale at each epoch:

| Component | Detail |
|-----------|--------|
| State     | `[current_loss, gradient_norm, weight_norm]` |
| Action    | Scalar ∈ (0, 1) → `effective_lr = base_lr × action` |
| Reward    | `prev_loss − curr_loss` (positive = improvement) |
| Policy    | Gaussian actor with shared trunk critic |

The PPO clip objective prevents destructively large policy updates:

```
L_CLIP = E[ min( r_t A_t,  clip(r_t, 1−ε, 1+ε) A_t ) ]
```

---

## Key design decisions

- **`torch.func.functional_call`** — MAML inner-loop uses functional forward passes so gradients flow through the update step without mutating model weights.
- **`create_graph=True`** in inner-loop `autograd.grad` — enables second-order meta-gradients for the outer update.
- **GAE advantages** — PPO uses Generalised Advantage Estimation (λ=0.95) for lower-variance policy gradients.
- **Entropy bonus** — small entropy coefficient (0.01) encourages exploration of lr schedules.
- **Task generalisation** — `generate_shifted_task()` applies a random spatial shift to class centres, creating out-of-distribution tasks for robustness evaluation.

---

## Experimental Results

| Method        | Normal Final Loss | Shifted Final Loss | Normal Threshold Epoch | Shifted Threshold Epoch |
|---------------|------------------|--------------------|------------------------|-------------------------|
| Baseline SGD  | 0.2736           | 0.2753             | 44.3 (3/5 tasks)       | 24.5 (2/5 tasks)        |
| MAML          | 0.1877           | 0.1319             | 20.0 (3/5 tasks)       | 20.6                    |
| MAML + PPO    | 0.2435           | 0.1647             | 21.0 (2/5 tasks)       | 14.7 (3/5 tasks)        |

---

## References

- Finn, C., Abbeel, P., & Levine, S. (2017). [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Neural Networks](https://arxiv.org/abs/1703.03400). ICML.
- Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv.
- Li, K., & Malik, J. (2017). [Learning to Optimize](https://arxiv.org/abs/1606.01885). ICLR.


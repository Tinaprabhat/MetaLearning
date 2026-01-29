

# 🧠 MetaLearning: Learning to Optimize Neural Networks

## Overview

This project explores **meta-learning** by building a system where a neural network learns **how to optimize another neural network**, rather than just learning task-specific weights.

Instead of manually tuning learning rates or relying on fixed optimization rules, a **meta-learner** observes training dynamics (loss, gradient norms, weight norms) and **adapts the learning strategy online**. The system is evaluated under **task distribution shift** to study when and why meta-learning helps.

> **Core idea:** Learn *learning behavior*, not just parameters.

---

## Motivation

Modern machine learning systems often struggle with:

* Distribution shift
* Noisy or unstable gradients
* Sensitivity to hyperparameters
* Manual optimizer tuning

Large-scale systems at FAANG companies invest heavily in **adaptive optimization** and **training efficiency**.
This project is a simplified but principled exploration of those ideas.

---

## Project Structure

```
MetaLearning/
│
├── tasks/                  # Synthetic task generators
│   ├── classification_tasks.py
│   └── task_generator.py
│
├── models/                 # Base learner (MLP)
│   └── base_learner.py
│
├── meta_learner/           # Meta-optimizer (learning to optimize)
│   └── meta_optimizer.py
│
├── experiments/            # Training, evaluation, and analysis
│   ├── training_monitor.py
│   ├── train_base_learner.py
│   ├── train_with_meta_learner.py
│   └── evaluate_task_shift.py
│
├── plots/                  # Generated evaluation plots
├── README.md
└── requirements.txt
```

The codebase follows a **research-style modular layout**, separating:

* task generation
* model definition
* meta-learning logic
* evaluation methodology

---

## System Components

### 1. Task Generator

* Generates synthetic classification tasks on the fly
* Supports **distribution shift** via shifted input distributions
* Ensures the system trains on a *distribution of problems*, not a single dataset

### 2. Base Learner

* Simple multi-layer perceptron (MLP)
* Trained with standard SGD
* Acts as a controlled baseline for comparison

### 3. Training Observability

During training, the system tracks:

* Loss trajectory
* Gradient norm
* Weight norm

These signals form the **state** observed by the meta-learner.

### 4. Meta-Learner (Meta-Optimizer)

* A small neural network that observes training state
* Outputs a **dynamic learning-rate scaling factor**
* Trained using reward based on loss improvement
* Can be **frozen and transferred** to new tasks

### 5. Evaluation Under Distribution Shift

* Meta-learner is trained on Task A
* Meta-learner is frozen
* Both baseline SGD and meta-optimizer are evaluated on **shifted Task B**
* Performance is compared using **epochs-to-threshold** metrics

---

## Experimental Findings (Honest Results)

Key observations from experiments:

* On **simple, well-tuned tasks**, fixed SGD often converges faster
* Under **distribution shift**, the meta-learner:

  * Maintains more stable gradient behavior
  * Adapts learning rates dynamically
  * Transfers optimization behavior across tasks
* Meta-learning does **not universally outperform SGD**, which is expected and consistent with literature

> **Conclusion:**
> Meta-learning is most valuable when optimization assumptions break, not when tasks are trivial.

This project emphasizes **correct experimental methodology and honest analysis**, rather than chasing artificial wins.

---

## Why This Project Matters

This work demonstrates:

* Understanding of **learning dynamics**, not just model accuracy
* Ability to design **fair and reproducible experiments**
* Awareness of **when meta-learning helps and when it doesn’t**
* Research-oriented ML engineering mindset

These skills are directly relevant to:

* ML systems engineering
* Research engineering roles
* Large-scale model training and optimization

---

## Tech Stack

* Python 3
* PyTorch
* NumPy
* Matplotlib

Designed to run on **CPU-only environments** (no GPU required).

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run task-shift evaluation:

```bash
python -m experiments.evaluate_task_shift
```

---

## Future Extensions

Possible next steps:

* Multi-task meta-training across many task instances
* Reinforcement learning–based meta-optimizers
* Lightweight architecture adaptation (self-evolving networks)
* Evaluation on real-world noisy datasets

---

## Key Takeaway

> This project is not about beating SGD at all costs.
> It is about understanding **how learning itself can be learned**, evaluated, and transferred.

That perspective is central to modern ML research and large-scale AI systems.

---


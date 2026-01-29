import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# experiments/train_with_meta_learner.py


from tasks.task_generator import generate_task, generate_shifted_task
from models.base_learner import BaseLearner
from meta_learners.meta_optimizer import MetaOptimizer
from experiments.training_monitor import (
    compute_gradient_norm,
    compute_weight_norm
)


def train_with_meta(
    epochs=50,
    base_lr=0.01,
    seed=42,
    loss_threshold=0.62,
    shifted=False,
    freeze_meta=False,
    meta_model=None
):
    torch.manual_seed(seed)

    if shifted:
        X, y = generate_shifted_task(seed)
    else:
        X, y = generate_task(seed=seed)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)


    base_model = BaseLearner()

    if meta_model is None:
        meta_model = MetaOptimizer()

    if freeze_meta:
        for p in meta_model.parameters():
            p.requires_grad = False

    base_optimizer = optim.SGD(base_model.parameters(), lr=base_lr)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    losses = []
    grad_norms = []
    lr_history = []
    threshold_epoch = None

    prev_loss = None

    for epoch in range(epochs):
        base_optimizer.zero_grad()

        out = base_model(X)
        loss = criterion(out, y)
        loss.backward()

        grad_norm = compute_gradient_norm(base_model)
        weight_norm = compute_weight_norm(base_model)

        state = torch.tensor(
            [[loss.item(), grad_norm, weight_norm]],
            dtype=torch.float32
        )

        lr_scale = meta_model(state)
        adjusted_lr = base_lr * lr_scale.item()

        for g in base_optimizer.param_groups:
            g["lr"] = adjusted_lr

        base_optimizer.step()

        # ---- Meta update ONLY if not frozen ----
        if prev_loss is not None and not freeze_meta:
            reward = prev_loss - loss.item()
            meta_loss = -reward * lr_scale.mean()

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        prev_loss = loss.item()

        losses.append(loss.item())
        grad_norms.append(grad_norm)
        lr_history.append(adjusted_lr)

        if threshold_epoch is None and loss.item() <= loss_threshold:
            threshold_epoch = epoch

    return {
        "loss": losses,
        "grad_norm": grad_norms,
        "lr": lr_history,
        "threshold_epoch": threshold_epoch,
        "meta_model": meta_model
    }

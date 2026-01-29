

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# experiments/train_base_learner.py

import torch
import torch.nn as nn
import torch.optim as optim

from tasks.task_generator import generate_task, generate_shifted_task
from models.base_learner import BaseLearner
from experiments.training_monitor import compute_gradient_norm


def train_base_model(
    epochs=50,
    lr=0.01,
    seed=42,
    loss_threshold=0.62,
    shifted=False
):
    torch.manual_seed(seed)

    if shifted:
        X, y = generate_shifted_task(seed)
    else:
        X, y = generate_task(seed=seed)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)


    model = BaseLearner()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    grad_norms = []
    threshold_epoch = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(X)
        loss = criterion(out, y)
        loss.backward()

        grad_norm = compute_gradient_norm(model)
        optimizer.step()

        losses.append(loss.item())
        grad_norms.append(grad_norm)

        if threshold_epoch is None and loss.item() <= loss_threshold:
            threshold_epoch = epoch

    return {
        "loss": losses,
        "grad_norm": grad_norms,
        "threshold_epoch": threshold_epoch
    }
